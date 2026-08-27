#!/bin/bash

# cd ~/cs336/assignments/assignment1-basics/cs336_basics


# cd utils

# uv run dataload.py 32000

# cd ..

# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.95 GiB. GPU 0 has a total capacity of 10.57 GiB of which 1.55 GiB is free. Including non-PyTorch memory, this process has 9.01 GiB memory in use. Of the allocated memory 8.76 GiB is allocated by PyTorch, and 65.22 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

# uv run train.py \
#         --experiment_name base_vs_32000 \
#         --train_data data/TinyStoriesV2-GPT4/vs_32000/token_ids/train/tokens.npy \
#         --val_data data/TinyStoriesV2-GPT4/vs_32000/token_ids/valid/tokens.npy \
#         --vocab_size 32000 \
#         \
#         --d_model 768 \
#         --num_layers 4 \
#         --num_heads 12 \
#         --context_len 256 \
#         --rope_theta 10000 \
#         --rms_eps 1e-5 \
#         \
#         --lr 1e-3 \
#         --beta1 0.9 \
#         --beta2 0.999 \
#         --opt_eps 1e-8 \
#         --weight_decay 1e-2 \
#         \
#         --lr_max 1e-3 \
#         --lr_min 1e-4 \
#         --warm_up 500 \
#         --cosine_end 7000 \
#         \
#         --max_iters 8000 \
#         --batch_size 64 \
#         --val_interval 100 \
#         --device cpu \
#         \
#         --save_dir runs \
#         --save_best_only \
#         --save_interval 1000 

# uv run train.py \
#         --experiment_name base_vs_32000 \
#         --train_data data/TinyStoriesV2-GPT4/vs_32000/token_ids/train/tokens.npy \
#         --val_data data/TinyStoriesV2-GPT4/vs_32000/token_ids/valid/tokens.npy \
#         --vocab_size 32000 \
#         \
#         --d_model 768 \
#         --num_layers 4 \
#         --num_heads 12 \
#         --context_len 256 \
#         --rope_theta 10000 \
#         --rms_eps 1e-5 \
#         \
#         --lr 1e-3 \
#         --beta1 0.9 \
#         --beta2 0.999 \
#         --opt_eps 1e-8 \
#         --weight_decay 1e-2 \
#         \
#         --lr_max 1e-3 \
#         --lr_min 1e-4 \
#         --warm_up 500 \
#         --cosine_end 7000 \
#         \
#         --max_iters 8000 \
#         --batch_size 32 \
#         --val_interval 100 \
#         --device cpu \
#         \
#         --save_dir runs \
#         --save_best_only \
#         --save_interval 1000 


uv run train.py \
        --experiment_name base_vs_32000_v2 \
        --train_data data/TinyStoriesV2-GPT4/vs_32000/token_ids/train/tokens.npy \
        --val_data data/TinyStoriesV2-GPT4/vs_32000/token_ids/valid/tokens.npy \
        --vocab_size 32000 \
        \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 16 \
        --context_len 128 \
        --rope_theta 10000 \
        --rms_eps 1e-5 \
        \
        --lr 1e-3 \
        --beta1 0.9 \
        --beta2 0.999 \
        --opt_eps 1e-8 \
        --weight_decay 1e-2 \
        \
        --lr_max 1e-3 \
        --lr_min 1e-4 \
        --warm_up 500 \
        --cosine_end 7000 \
        \
        --max_iters 8000 \
        --batch_size 64 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000