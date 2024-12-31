#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from constant import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAMES,
    TRAIN_LABEL_NAMES
)

from tqdm import tqdm
from lr_scheduler import LRPolicyScheduler
import pandas as pd
from models import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from binary_dataloader import BinaryDataloader
from typing import Tuple, Optional
import torchsnapshot

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.

class InteractionType(Enum):
    ORIGINAL = "original"
    DCN = "dcn"
    PROJECTION = "projection"

    def __str__(self):
        return self.value


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop the last non-full training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["dnn", "dot_att"],
        default="dnn",
        help="model type",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["criteo_1t", "criteo_kaggle"],
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=1000000000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=323,
        help="Num of features",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=16,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--interaction_branch1_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch1 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--interaction_branch2_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch2 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--binary_path",
        type=str,
        default=None,
        help="Directory path containing the Criteo dataset npy files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the model.",
    )
    parser.add_argument(
        "--training_days",
        type=str,
        default="14, 15",
        help="Comma separated days.",
    )
    parser.add_argument(
        "--valid_hour",
        type=str,
        default="20/22",
        help="validation hour day 20 hour 22",
    )
    parser.add_argument(
        "--test_hour",
        type=str,
        default="20/23",
        help="validation hour day 20 hour 23",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for Adagrad optimizer.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle the training set in memory. This will override mmap_mode",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
        drop_last=None,
        shuffle_batches=None,
        shuffle_training_set=None,
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument(
        "--interaction_type",
        type=InteractionType,
        choices=list(InteractionType),
        default=InteractionType.ORIGINAL,
        help="Determine the interaction type to be used (original, dcn, or projection)"
        " default is original DLRM with pairwise dot product",
    )
    parser.add_argument(
        "--collect_multi_hot_freqs_stats",
        dest="collect_multi_hot_freqs_stats",
        action="store_true",
        help="Flag to determine whether to collect stats on freq of embedding access.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=None,
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default=None,
        help="Multi-hot distribution options.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_start", type=int, default=0)
    parser.add_argument("--lr_decay_steps", type=int, default=0)
    parser.add_argument(
        "--print_lr",
        action="store_true",
        help="Print learning rate every iteration.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    
    parser.add_argument(
        "--dataset_type", 
        type=str, 
        default="random",
        help="Dataset type: orc, pt, arrow, random, bin"
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Number of files to train.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="prefetch factor for each worker of dataloader.",
    )
    parser.add_argument("--snapshot_path", type=str, default=None)
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Only do test, not train.",
    )

    return parser.parse_args(argv)

def _evaluate(
    limit_batches: Optional[int],
    pipeline: TrainPipelineSparseDist,
    eval_dataloader: DataLoader,
    stage: str,
    save_dir: str = 'output',
) -> Tuple[float, float]:
    
    pipeline._model.eval()
    device = pipeline._device

    iterator = itertools.islice(iter(eval_dataloader), limit_batches)

    num_labels = len(TRAIN_LABEL_NAMES)
    if num_labels > 1:
        auroc = metrics.AUROC(task='multilabel', num_labels=num_labels, average=None).to(device)  
    else:
        auroc = metrics.AUROC(task='binary', num_labels=num_labels).to(device)

    all_preds = []
    all_labels = []

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_rank_zero = rank == 0

    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Evaluating {stage} set",
            total=len(eval_dataloader),
            disable=False,
        )

    sample_idx = 0
    indices = []
    batch_size = 1024

    with torch.no_grad():
        while True:
            try:
                _loss, preds, labels = pipeline.progress(iterator)
                preds = torch.sigmoid(preds) # here preds is logits, convert it to sigmoid
                auroc(preds, labels)
                all_preds.append(preds)
                all_labels.append(labels)
                batch_size = len(preds)
                indices.extend([i for i in range(sample_idx, sample_idx + len(preds))])
                sample_idx += len(preds) * world_size
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                break

    auroc_result = auroc.compute().cpu().numpy() # .item()

    # Concatenate local predictions and labels
    indices = [rank*batch_size+id for id in indices]
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    indices = torch.tensor(indices, device=device)

    # Get lengths of tensors from all ranks
    local_size = torch.tensor([len(all_preds)], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = torch.cat(all_sizes).tolist()

    # Gather predictions, labels and indices from all ranks
    gathered_preds = [torch.zeros((size, num_labels), dtype=all_preds.dtype, device=device) for size in all_sizes]
    gathered_labels = [torch.zeros((size, num_labels), dtype=all_labels.dtype, device=device) for size in all_sizes]
    gathered_indices = [torch.zeros(size, dtype=indices.dtype, device=device) for size in all_sizes]

    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_labels, all_labels)
    dist.all_gather(gathered_indices, indices)

    if is_rank_zero:
        # Combine all gathered data
        all_preds_gathered = torch.cat(gathered_preds).cpu().numpy()
        all_labels_gathered = torch.cat(gathered_labels).cpu().numpy()
        all_indices_gathered = torch.cat(gathered_indices).cpu().numpy()

        # Create and save combined dataframe
        df = pd.DataFrame({
                'original_index': all_indices_gathered
            })
        for idx, column in enumerate(TRAIN_LABEL_NAMES):
            df[f'{column}_pred'] = all_preds_gathered[:, idx]
            df[f'{column}_true'] = all_labels_gathered[:, idx]
        
        df = df.sort_values('original_index')
        df = df.drop('original_index', axis=1)
        
        save_path = os.path.join(save_dir, 'predict_results.csv')
        df.to_csv(save_path, index=False)
        print(f"Combined {len(df)} predictions saved to {save_path}")

    # Get the total number of samples across all ranks
    num_samples = torch.tensor(len(all_labels), device=device)
    print(f"rank: {rank}, num_samples: {num_samples}")
    dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)

    # Sum up predictions and labels across all ranks
    sum_preds = all_preds.sum(axis=0)
    sum_labels = all_labels.float().sum(axis=0)
    dist.all_reduce(sum_preds, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_labels, op=dist.ReduceOp.SUM)

    # Calculate PCOC using the aggregated data
    predicted_ctr = sum_preds / num_samples
    true_ctr = sum_labels / num_samples
    pcoc = predicted_ctr / torch.clamp(true_ctr, min=1e-7)
    pcoc = pcoc.cpu().numpy()

    if is_rank_zero:
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"PCOC over {stage} set: {pcoc}")
        print(f"Number of {stage} samples: {num_samples}")
    return auroc_result, pcoc

def batched(it: Iterator, n: int):
    assert n >= 1
    for x in it:
        yield itertools.chain((x,), itertools.islice(it, n - 1))

def _train(
    pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epoch: int,
    lr_scheduler,
    print_lr: bool,
    validation_freq: Optional[int],
    limit_train_batches: Optional[int],
    limit_val_batches: Optional[int],
) -> None:
    """
    Trains model for 1 epoch. Helper function for train_val_test.

    Args:
        pipeline (TrainPipelineSparseDist): data pipeline.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        epoch (int): The number of complete passes through the training set so far.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.
        print_lr (bool): Whether to print the learning rate every training step.
        validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
        limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
        limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.

    Returns:
        None.
    """
    pipeline._model.train()

    iterator = itertools.islice(iter(train_dataloader), limit_train_batches)

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
            disable=False,
        )

    start_it = 0
    n = (
        validation_freq
        if validation_freq
        else limit_train_batches if limit_train_batches else len(train_dataloader)
    )

    for batched_iterator in batched(iterator, n):
        for it in itertools.count(start_it):
            try:
                if is_rank_zero and print_lr:
                    for i, g in enumerate(pipeline._optimizer.param_groups):
                        print(f"lr: {it} {i} {g['lr']:.6f}")
                
                pipeline.progress(batched_iterator)
                lr_scheduler.step()
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                if is_rank_zero:
                    print("Total number of iterations:", it)
                start_it = it
                break

        if validation_freq and start_it % validation_freq == 0:
            _evaluate(limit_val_batches, pipeline, val_dataloader, "val")
            pipeline._model.train()

@dataclass
class TrainValTestResults:
    val_aurocs: List[float] = field(default_factory=list)
    val_pcocs: List[float] = field(default_factory=list)
    test_auroc: Optional[float] = None
    test_pcoc: Optional[float] = None


def train_val_test(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    lr_scheduler: LRPolicyScheduler,
) -> TrainValTestResults:
    """
    Train/validation/test loop.

    Args:
        args (argparse.Namespace): parsed command line args.
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        test_dataloader (DataLoader): Test set's dataloader.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.

    Returns:
        TrainValTestResults.
    """
    os.environ["TORCHSNAPSHOT_ENABLE_SHARDED_TENSOR_ELASTICITY_ROOT_ONLY"] = "1"
    progress = torchsnapshot.StateDict(current_epoch=0)

    # torchsnapshot: define app state
    app_state = {
        "dmp": model,
        "optim": model.fused_optimizer,
        "progress": progress,
    }

    # torchsnapshot: restore from snapshot
    snapshot_path = args.snapshot_path
    if snapshot_path is not None:
        snapshots = {}
        for k, v in app_state.items():
            # snapshots[k] = torchsnapshot.Snapshot.take(
            #     path=f"{args.save_dir}/epoch_{epoch}/{k}",
            #     app_state={k: v},
            #     replicated=["**"],
            #     )
            
            snapshot = torchsnapshot.Snapshot(
                path=f"{snapshot_path}/{k}",
            )
            snapshot.restore(app_state={k: v})

        print(f"Restored snapshot from {snapshot_path}.")

    results = TrainValTestResults()
    pipeline = TrainPipelineSparseDist(
        model, optimizer, device, execute_all_batches=True
    )

    if not args.test_mode:
        for epoch in range(args.epochs):
            _train(
                pipeline,
                train_dataloader,
                val_dataloader,
                epoch,
                lr_scheduler,
                args.print_lr,
                args.validation_freq_within_epoch,
                args.limit_train_batches,
                args.limit_val_batches,
            )
            # val_auroc, val_pcoc = _evaluate(args.limit_val_batches, pipeline, val_dataloader, "val")
            # results.val_aurocs.append(val_auroc)
            # results.val_aurocs.append(val_pcoc)

            # # torchsnapshot: take snapshot
            # snapshot = torchsnapshot.Snapshot.take(
            #     path=f"{args.save_dir}/epoch_{epoch}",
            #     app_state=app_state,
            #     replicated=["**"],
            # )

            # print(f"Snapshot path: {snapshot.path}")

            snapshots = {}
            for k, v in app_state.items():
                snapshots[k] = torchsnapshot.Snapshot.take(
                    path=f"{args.save_dir}/epoch_{epoch}/{k}",
                    app_state={k: v},
                    replicated=["**"],
                )

            progress["current_epoch"] += 1

    test_auroc, test_pcoc = _evaluate(args.limit_test_batches, pipeline, test_dataloader, "test", args.save_dir)
    results.test_auroc = test_auroc
    results.test_pcoc = test_pcoc

    # torchsnapshot: examine snapshot content
    if dist.get_rank() == 0:
        for k, snapshot in snapshots.items():
            entries = snapshot.get_manifest()
            for path in entries.keys():
                print(path)      

    return results


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """
    args = parse_args(argv)
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    if args.multi_hot_sizes is not None:
        assert (
            args.num_embeddings_per_feature is not None
            and len(args.multi_hot_sizes) == len(args.num_embeddings_per_feature)
            or args.num_embeddings_per_feature is None
            and len(args.multi_hot_sizes) == len(DEFAULT_CAT_NAMES)
        ), "--multi_hot_sizes must be a comma delimited list the same size as the number of embedding tables."
    assert (
        args.binary_path is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--binary_path and --synthetic_multi_hot_criteo_path are mutually exclusive CLI arguments."
    assert (
        args.multi_hot_sizes is None or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_sizes is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."
    assert (
        args.multi_hot_distribution_type is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_distribution_type is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(
            "PARAMS: (lr, batch_size, warmup_steps, decay_start, decay_steps): "
            f"{(args.learning_rate, args.batch_size, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps)}"
        )

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings = None

    # Sets default limits for random dataloader iterations when left unspecified.
    if (
        args.binary_path
        is args.synthetic_multi_hot_criteo_path
        is None
    ):
        for split in ["train", "val", "test"]:
            attr = f"limit_{split}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    train_path = os.path.join(args.binary_path, 'train')
    train_dataloader = BinaryDataloader(
        binary_file_path=train_path,
        batch_size=args.batch_size,
    ).get_dataloader(rank=rank, world_size=world_size, shuffle=args.shuffle_training_set, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    valid_path = os.path.join(args.binary_path, 'valid')
    val_dataloader = BinaryDataloader(
        binary_file_path=valid_path,
        batch_size=args.batch_size,
    ).get_dataloader(rank=rank, world_size=world_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    test_path = os.path.join(args.binary_path, 'valid')
    test_dataloader = BinaryDataloader(
        binary_file_path=test_path,
        batch_size=args.batch_size,
    ).get_dataloader(rank=rank, world_size=world_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    
    print(f"rank: {rank}, train_dataloader: {len(train_dataloader)}, test_dataloader: {len(test_dataloader)}")
    
    eb_configs = [
        EmbeddingBagConfig(
            name="shared_embedding",
            embedding_dim=args.embedding_dim,
            num_embeddings=(
                none_throws(args.num_embeddings_per_feature)
                if args.num_embeddings is None
                else args.num_embeddings
            ),
            feature_names=DEFAULT_CAT_NAMES,
        )
    ]

    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = args.over_arch_layer_sizes

    if args.interaction_type == InteractionType.ORIGINAL:
        dlrm_model = DLRM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dense_device=device,
        )
    elif args.interaction_type == InteractionType.DCN:
        dlrm_model = DLRM_DCN(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dcn_num_layers=args.dcn_num_layers,
            dcn_low_rank_dim=args.dcn_low_rank_dim,
            dense_device=device,
        )
    elif args.interaction_type == InteractionType.PROJECTION:
        dlrm_model = DLRM_Projection(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            interaction_branch1_layer_sizes=args.interaction_branch1_layer_sizes,
            interaction_branch2_layer_sizes=args.interaction_branch2_layer_sizes,
            dense_device=device,
        )
    else:
        raise ValueError(
            "Unknown interaction option set. Should be original, dcn, or projection."
        )

    train_model = DLRMTrain(dlrm_model)
    embedding_optimizer = torch.optim.Adagrad if args.adagrad else torch.optim.SGD
    # This will apply the Adagrad optimizer in the backward pass for the embeddings (sparse_arch). This means that
    # the optimizer update will be applied in the backward pass, in this case through a fused op.
    # TorchRec will use the FBGEMM implementation of EXACT_ADAGRAD. For GPU devices, a fused CUDA kernel is invoked. For CPU, FBGEMM_GPU invokes CPU kernels
    # https://github.com/pytorch/FBGEMM/blob/2cb8b0dff3e67f9a009c4299defbd6b99cc12b8f/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L676-L678

    # Note that lr_decay, weight_decay and initial_accumulator_value for Adagrad optimizer in FBGEMM v0.3.2
    # cannot be specified below. This equivalently means that all these parameters are hardcoded to zero.
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.adagrad:
        optimizer_kwargs["eps"] = args.eps
    apply_optimizer_in_backward(
        embedding_optimizer,
        train_model.model.sparse_arch.parameters(),
        optimizer_kwargs,
    )

    constraints = {
        "shared_embedding": ParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],  # ROW_WISE
        )
    }
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        constraints=constraints,
        batch_size=args.batch_size,
        # If experience OOM, increase the percentage. see
        # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )
    plan = planner.collective_plan(
        train_model, get_default_sharders(), dist.GroupMember.WORLD
    )

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        plan=plan,
    )
    if rank == 0:
        print('model:', model)
    if rank == 0 and args.print_sharding_plan:
        for collectionkey, plans in model._plan.plan.items():
            print(collectionkey)
            for table_name, plan in plans.items():
                print(table_name, "\n", plan, "\n")

    def optimizer_with_params():
        if args.adagrad:
            return lambda params: torch.optim.Adagrad(
                params, lr=args.learning_rate, eps=args.eps
            )
        else:
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)

    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        optimizer_with_params(),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    lr_scheduler = LRPolicyScheduler(
        optimizer, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps
    )

    train_val_test(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        lr_scheduler,
    )

def invoke_main() -> None:
    main(sys.argv[1:])

if __name__ == "__main__":
    invoke_main()
