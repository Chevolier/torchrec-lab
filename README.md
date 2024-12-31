This project uses torchrec for recommendation.

This README shows how to prepare data and train using torchrec.

# Prepare Environments
```bash
conda create -n torchrec python=3.11 -y
conda activate torchrec

# For CUDA < 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics==1.0.3
pip install torchrec --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

# Data Process

```bash
nohup python -u process_data.py \
    --data_path datasets/data.txt \
    --num_embeddings 1000000 \
    --num_workers 48 \
    --output_dir datasets/v0 \
    > logs/data_proc.out 2>&1 &
```

# Train

This allows training using 1 or multiple labels, to do this, change TRAIN_LABEL_NAMES in constant.py and make the last value in --over_arch_layer_sizes "512,512,256,2" to be the same with len(TRAIN_LABEL_NAMES), i.e., 2 = len(TRAIN_LABEL_NAMES).

```bash
# train
torchx run -s local_cwd dist.ddp -j 1x1 --script dlrm_main.py -- --model_type dnn --print_sharding_plan --epochs 1 --embedding_dim 16 --dense_arch_layer_sizes "512,256,16" --over_arch_layer_sizes "512,512,256,2" --batch_size 8192 --learning_rate 0.001 --adagrad --num_embeddings 10000000 --binary_path ./datasets/v0 --num_workers 4 --prefetch_factor 4 --save_dir checkpoint 

nohup time torchx run -s local_cwd dist.ddp -j 1x1 --script dlrm_main.py -- --model_type dnn --print_sharding_plan --epochs 1 --embedding_dim 16 --dense_arch_layer_sizes "512,256,16" --over_arch_layer_sizes "512,512,256,2" --batch_size 8192 --learning_rate 0.001 --adagrad --num_embeddings 10000000 --binary_path ./datasets/v0 --num_workers 4 --prefetch_factor 4 --save_dir checkpoint > logs/train.out 2>&1 &

# only test, no train
nohup time torchx run -s local_cwd dist.ddp -j 1x1 --script dlrm_main.py -- --model_type dnn --print_sharding_plan --epochs 1 --embedding_dim 16 --dense_arch_layer_sizes "512,256,16" --over_arch_layer_sizes "512,512,256,2" --batch_size 8192 --learning_rate 0.001 --adagrad --num_embeddings 10000000 --binary_path ./datasets/v0 --num_workers 4 --prefetch_factor 4 --save_dir checkpoint --snapshot_path './checkpoint/epoch_0' --test_mode > logs/test.out 2>&1 &

```

