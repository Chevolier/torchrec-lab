#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from torch.utils import data as data_utils
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from constant import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAMES,
    TRAIN_LABEL_NAMES
)
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class ParametricDataset(Dataset):
    def __init__(
        self,
        binary_file_path: str,
        batch_size: int,
        drop_last_batch,
        **kwargs,
    ):
        self._batch_size = batch_size
        bytes_per_feature = {}
        for name in DEFAULT_INT_NAMES:
            bytes_per_feature[name] = np.dtype(np.float32).itemsize
        for name in DEFAULT_CAT_NAMES:
            bytes_per_feature[name] = np.dtype(np.int32).itemsize

        self._numerical_features_file = None
        self._label_file = None
        self._categorical_features_files = []

        self._numerical_bytes_per_batch = (
            bytes_per_feature[DEFAULT_INT_NAMES[0]]
            * len(DEFAULT_INT_NAMES)
            * batch_size
        )
        self._label_bytes_per_batch = np.dtype(np.float32).itemsize * batch_size
        self._categorical_bytes_per_batch = [
            bytes_per_feature[feature] * self._batch_size
            for feature in DEFAULT_CAT_NAMES
        ]
        # Load categorical
        for feature_name in DEFAULT_CAT_NAMES:
            path_to_open = os.path.join(binary_file_path, f"{feature_name}.bin")
            cat_file = os.open(path_to_open, os.O_RDONLY)
            bytes_per_batch = bytes_per_feature[feature_name] * self._batch_size
            batch_num_float = os.fstat(cat_file).st_size / bytes_per_batch
            self._categorical_features_files.append(cat_file)

        # Load numerical
        path_to_open = os.path.join(binary_file_path, "numerical.bin")
        self._numerical_features_file = os.open(path_to_open, os.O_RDONLY)
        batch_num_float = (
            os.fstat(self._numerical_features_file).st_size
            / self._numerical_bytes_per_batch
        )

        # Load label
        path_to_open = os.path.join(binary_file_path, "label.bin")
        self._label_file = os.open(path_to_open, os.O_RDONLY)
        batch_num_float = (
            os.fstat(self._label_file).st_size / self._label_bytes_per_batch
        )
        # number of batches means the ALL data for all ranks
        number_of_batches = (
            math.ceil(batch_num_float)
            if not drop_last_batch
            else math.floor(batch_num_float)
        )
        # for this data_loader, we should divide the num_batch by world_size
        self._num_entries = number_of_batches

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx: int):
        """Numerical features are returned in the order they appear in the channel spec section
        For performance reasons, this is required to be the order they are saved in, as specified
        by the relevant chunk in source spec.
        Categorical features are returned in the order they appear in the channel spec section
        """

        if idx >= self._num_entries:
            raise IndexError()

        return self._get_item(idx)

    def _get_item(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        click = self._get_label(idx)
        numerical_features = self._get_numerical_features(idx)
        categorical_features = self._get_categorical_features(idx)

        return numerical_features, categorical_features, click

    def _get_label(self, idx: int) -> torch.Tensor:
        raw_label_data = os.pread(
            self._label_file,
            self._label_bytes_per_batch,
            idx * self._label_bytes_per_batch,
        )
        array = np.frombuffer(raw_label_data, dtype=np.float32)
        return torch.from_numpy(array).to(torch.float32)

    def _get_numerical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._numerical_features_file is None:
            return None

        raw_numerical_data = os.pread(
            self._numerical_features_file,
            self._numerical_bytes_per_batch,
            idx * self._numerical_bytes_per_batch,
        )
        array = np.frombuffer(raw_numerical_data, dtype=np.float32)
        return (
            torch.from_numpy(array).to(torch.float32).view(-1, len(DEFAULT_INT_NAMES))
        )

    def _get_categorical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._categorical_features_files is None:
            return None
        categorical_features = []
        for cat_bytes, cat_file in zip(
            self._categorical_bytes_per_batch,
            self._categorical_features_files,
        ):
            raw_cat_data = os.pread(cat_file, cat_bytes, idx * cat_bytes)
            array = np.frombuffer(raw_cat_data, dtype=np.int32)
            tensor = torch.from_numpy(array).to(torch.long).view(-1)
            categorical_features.append(tensor)
        return torch.cat(categorical_features)

class VarParametricDataset(Dataset):  # Var stands for variable length
    def __init__(
        self,
        binary_file_path: str,
        batch_size: int,
        drop_last_batch,
        **kwargs,
    ):
        self._batch_size = batch_size

        # Load categorical
        self._num_categorical_features = len(DEFAULT_CAT_NAMES)
        self._label_file = None
        self._categorical_feature_file = None
        self._categorical_length_file = None
        self._categorical_cum_length_file = None
        
        self._length_bytes_per_batch = np.dtype(np.int32).itemsize * batch_size
        self._cum_length_bytes_per_batch = np.dtype(np.int64).itemsize * batch_size

        self._feature_bytes_per_item = np.dtype(np.int64).itemsize

        path_to_open = os.path.join(binary_file_path, f"cat_value.bin")
        self._categorical_feature_file = os.open(path_to_open, os.O_RDONLY)

        path_to_open = os.path.join(binary_file_path, f"cat_length.bin")
        self._categorical_length_file = os.open(path_to_open, os.O_RDONLY)

        path_to_open = os.path.join(binary_file_path, f"cat_cum_length.bin")
        self._categorical_cum_length_file = os.open(path_to_open, os.O_RDONLY)

        # Load numerical
        bytes_per_feature = {}
        for name in DEFAULT_INT_NAMES:
            bytes_per_feature[name] = np.dtype(np.float32).itemsize

        self._numerical_bytes_per_batch = (
            bytes_per_feature[DEFAULT_INT_NAMES[0]]
            * len(DEFAULT_INT_NAMES)
            * batch_size
        )

        path_to_open = os.path.join(binary_file_path, "numerical.bin")
        self._numerical_features_file = os.open(path_to_open, os.O_RDONLY)
        batch_num_float = (
            os.fstat(self._numerical_features_file).st_size
            / self._numerical_bytes_per_batch
        )

        # Load labels
        self._label_bytes_per_batch = np.dtype(np.int32).itemsize * len(DEFAULT_LABEL_NAMES) * batch_size
        path_to_open = os.path.join(binary_file_path, "label.bin")
        self._label_file = os.open(path_to_open, os.O_RDONLY)

        batch_num_float = (
            os.fstat(self._label_file).st_size / self._label_bytes_per_batch
        )

        # number of bytes per feature for cum_length and length
        self._cum_length_bytes_per_feature = os.fstat(self._categorical_cum_length_file).st_size // self._num_categorical_features
        self._length_bytes_per_feature = os.fstat(self._categorical_length_file).st_size // self._num_categorical_features

        # number of batches means the ALL data for all ranks
        number_of_batches = (
            math.ceil(batch_num_float)
            if not drop_last_batch
            else math.floor(batch_num_float)
        )
        # for this data_loader, we should divide the num_batch by world_size
        self._num_entries = number_of_batches

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx: int):
        """Numerical features are returned in the order they appear in the channel spec section
        For performance reasons, this is required to be the order they are saved in, as specified
        by the relevant chunk in source spec.
        Categorical features are returned in the order they appear in the channel spec section
        """

        if idx >= self._num_entries:
            raise IndexError()

        return self._get_item(idx)

    def _get_item(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        click = self._get_label(idx)
        numerical_features = self._get_numerical_features(idx)
        categorical_features, lengths = self._get_categorical_features(idx)

        return categorical_features, lengths, numerical_features, click

    def _get_label(self, idx: int) -> torch.Tensor:
        raw_label_data = os.pread(
            self._label_file,
            self._label_bytes_per_batch,
            idx * self._label_bytes_per_batch,
        )
        array = np.frombuffer(raw_label_data, dtype=np.int32)
        all_labels = torch.tensor(array, dtype=torch.int32).view(-1, len(DEFAULT_LABEL_NAMES))
        train_label_ids = [id for id, label in enumerate(DEFAULT_LABEL_NAMES) if label in TRAIN_LABEL_NAMES]
        train_labels = all_labels[:, train_label_ids]
        return train_labels

    def _get_numerical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._numerical_features_file is None:
            return None

        raw_numerical_data = os.pread(
            self._numerical_features_file,
            self._numerical_bytes_per_batch,
            idx * self._numerical_bytes_per_batch,
        )
        array = np.frombuffer(raw_numerical_data, dtype=np.float32)
        return (
            torch.from_numpy(array).to(torch.float32).view(-1, len(DEFAULT_INT_NAMES))
        )

    def _get_categorical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._categorical_feature_file is None:
            return None

        categorical_features = []
        lengths = []
        
        for feature_id in range(self._num_categorical_features):
            cum_length_data = os.pread(
                self._categorical_cum_length_file,
                self._cum_length_bytes_per_batch,
                feature_id * self._cum_length_bytes_per_feature + idx * self._cum_length_bytes_per_batch,
            )
            cum_length = np.frombuffer(cum_length_data, dtype=np.int64)

            length_data = os.pread(
                self._categorical_length_file,
                self._length_bytes_per_batch,
                feature_id * self._length_bytes_per_feature + idx * self._length_bytes_per_batch,
            )
            length = np.frombuffer(length_data, dtype=np.int32)

            feature_offset = (cum_length[0]-length[0]) * self._feature_bytes_per_item
            feature_bytes_batch = (cum_length[-1]-(cum_length[0]-length[0])) * self._feature_bytes_per_item

            raw_cat_data = os.pread(self._categorical_feature_file, feature_bytes_batch, feature_offset)
            feature = np.frombuffer(raw_cat_data, dtype=np.int64)

            length = torch.tensor(length, dtype=torch.int32).view(-1)
            lengths.append(length)

            feature = torch.tensor(feature, dtype=torch.int64).view(-1)
            categorical_features.append(feature)
        
        lengths = torch.cat(lengths)
        categorical_features = torch.cat(categorical_features)

        return categorical_features, lengths

class BinaryDataloader:
    def __init__(
        self,
        binary_file_path: str | List[str],
        batch_size: int = 2048,
        drop_last_batch: bool = True,  # the last batch may not contain enough data which breaks the size of KJT
    ) -> None:
        if isinstance(binary_file_path, str):
            self.dataset = VarParametricDataset(
                binary_file_path,
                batch_size,
                drop_last_batch,
            )
        elif isinstance(binary_file_path, list):
            dataset_list = [
                VarParametricDataset(
                    path,
                    batch_size,
                    drop_last_batch,
                )
                for path in binary_file_path
            ]
            self.dataset = ConcatDataset(dataset_list)
        else:
            raise TypeError("binary_file_path must be a string or a list of strings")

        self.keys: List[str] = DEFAULT_CAT_NAMES
        num_cat_features = len(DEFAULT_CAT_NAMES)
        self.stride = batch_size
        self.length_per_key: List[int] = num_cat_features * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(num_cat_features + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def collate_fn(self, attr_dict):
        sparse_features, sparse_lengths, dense_features, labels = attr_dict
        return Batch(
            dense_features=dense_features,
            sparse_features=KeyedJaggedTensor(
                keys=DEFAULT_CAT_NAMES,
                values=sparse_features,
                lengths=sparse_lengths,
            ),
            labels=labels,
        )

    def get_dataloader(
        self,
        rank: int,
        world_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = None,
    ) -> data_utils.DataLoader:
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        dataloader = data_utils.DataLoader(
            self.dataset,
            batch_size=None,
            pin_memory=True,
            collate_fn=self.collate_fn,
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        return dataloader
