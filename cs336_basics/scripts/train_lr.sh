#!/bin/bash

# cd ~/cs336/assignments/assignment1-basics/cs336_basics

# lr_max: 1e-4, 1e-3, 1e-2, 1e-1, 1
# lr_min: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1

uv run train.py \
        --experiment_name LR_small \
        --train_data data/TinyStoriesV2-GPT4/vs_10000/token_ids/train/tokens.npy \
        --val_data data/TinyStoriesV2-GPT4/vs_10000/token_ids/valid/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 16 \
        --context_len 256 \
        --rope_theta 10000 \
        --rms_eps 1e-5 \
        \
        --lr 1e-3 \
        --beta1 0.9 \
        --beta2 0.999 \
        --opt_eps 1e-8 \
        --weight_decay 1e-2 \
        \
        --lr_max 1e-4 \
        --lr_min 1e-5 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 64 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000 


uv run train.py \
        --experiment_name LR_large \
        --train_data data/TinyStoriesV2-GPT4/vs_10000/token_ids/train/tokens.npy \
        --val_data data/TinyStoriesV2-GPT4/vs_10000/token_ids/valid/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 16 \
        --context_len 256 \
        --rope_theta 10000 \
        --rms_eps 1e-5 \
        \
        --lr 1e-3 \
        --beta1 0.9 \
        --beta2 0.999 \
        --opt_eps 1e-8 \
        --weight_decay 1e-2 \
        \
        --lr_max 1e-2 \
        --lr_min 1e-3 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 64 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000 


uv run train.py \
        --experiment_name LR_larger \
        --train_data data/TinyStoriesV2-GPT4/vs_10000/token_ids/train/tokens.npy \
        --val_data data/TinyStoriesV2-GPT4/vs_10000/token_ids/valid/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 16 \
        --context_len 256 \
        --rope_theta 10000 \
        --rms_eps 1e-5 \
        \
        --lr 1e-3 \
        --beta1 0.9 \
        --beta2 0.999 \
        --opt_eps 1e-8 \
        --weight_decay 1e-2 \
        \
        --lr_max 1e-1 \
        --lr_min 1e-2 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 64 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000 


uv run train.py \
        --experiment_name LR_x_large \
        --train_data data/TinyStoriesV2-GPT4/vs_10000/token_ids/train/tokens.npy \
        --val_data data/TinyStoriesV2-GPT4/vs_10000/token_ids/valid/tokens.npy \
        --vocab_size 10000 \
        \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 16 \
        --context_len 256 \
        --rope_theta 10000 \
        --rms_eps 1e-5 \
        \
        --lr 1e-3 \
        --beta1 0.9 \
        --beta2 0.999 \
        --opt_eps 1e-8 \
        --weight_decay 1e-2 \
        \
        --lr_max 1 \
        --lr_min 1e-1 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 64 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000