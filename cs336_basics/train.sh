#!/bin/bash

# cd ~/cs336/assignments/assignment1-basics

# uv run train.py \
#         --train_data data/tokens/TinyStoriesV2-GPT4/train/tokens.npy \
#         --val_data data/tokens/TinyStoriesV2-GPT4/tokens.npy \
#         --vocab_size 10000 \
#         \
#         --d_model 768 \
#         --num_layers 12 \
#         --num_heads 12 \
#         --context_len 1024 \
#         --rope_theta 1e5 \
#         --rms_eps 1e-5 \
#         \
#         --lr 1e-3 \
#         --beta1 0.9 \
#         --beta2 0.999 \
#         --opt_eps 1e-8 \
#         --weight_decay 1e-2 \
#         \
#         --lr_max 6e-4 \
#         --lr_min 6e-5 \
#         --warmup 2000 \
#         --cosine_end 48000 \
#         \
#         --max_iters 5e4 \
#         --batch_size 16 \
#         --val_interval 500 \
#         --device cpu \
#         \
#         --save_dir checkpoints \
#         --save_interval 1000 \

# cpu上运行

uv run train.py \
        --train_data data/tokens/TinyStoriesV2-GPT4/train/tokens.npy \
        --val_data data/tokens/TinyStoriesV2-GPT4/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 384 \
        --num_layers 6 \
        --num_heads 6 \
        --context_len 128 \
        --rope_theta 1e5 \
        --rms_eps 1e-5 \
        \
        --lr 1e-3 \
        --beta1 0.9 \
        --beta2 0.999 \
        --opt_eps 1e-8 \
        --weight_decay 1e-2 \
        \
        --lr_max 3e-4 \
        --lr_min 3e-5 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 4 \
        --val_interval 500 \
        --device cpu \
        \
        --save_dir checkpoints \
        --save_best_only \
        --save_interval 1000 