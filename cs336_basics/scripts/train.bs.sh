#!/bin/bash

# cd ~/cs336/assignments/assignment1-basics/cs336_basics

# batch size: 1, 8, 16, 32

uv run train.py \
        --train_data data/tokens/TinyStoriesV2-GPT4/train/tokens.npy \
        --val_data data/tokens/TinyStoriesV2-GPT4/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 256 \
        --num_layers 6 \
        --num_heads 16 \
        --context_len 96 \
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
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 1 \
        --val_interval 500 \
        --device cpu \
        \
        --save_dir checkpoints \
        --save_best_only \
        --save_interval 1000


uv run train.py \
        --train_data data/tokens/TinyStoriesV2-GPT4/train/tokens.npy \
        --val_data data/tokens/TinyStoriesV2-GPT4/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 256 \
        --num_layers 6 \
        --num_heads 16 \
        --context_len 96 \
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
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 8 \
        --val_interval 500 \
        --device cpu \
        \
        --save_dir checkpoints \
        --save_best_only \
        --save_interval 1000 


uv run train.py \
        --train_data data/tokens/TinyStoriesV2-GPT4/train/tokens.npy \
        --val_data data/tokens/TinyStoriesV2-GPT4/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 256 \
        --num_layers 6 \
        --num_heads 16 \
        --context_len 96 \
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
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 32 \
        --val_interval 500 \
        --device cpu \
        \
        --save_dir checkpoints \
        --save_best_only \
        --save_interval 1000 

uv run train.py \
        --train_data data/tokens/TinyStoriesV2-GPT4/train/tokens.npy \
        --val_data data/tokens/TinyStoriesV2-GPT4/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 256 \
        --num_layers 6 \
        --num_heads 16 \
        --context_len 96 \
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
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 64 \
        --val_interval 500 \
        --device cpu \
        \
        --save_dir checkpoints \
        --save_best_only \
        --save_interval 1000 

