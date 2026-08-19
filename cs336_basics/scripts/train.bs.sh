#!/bin/bash

# cd ~/cs336/assignments/assignment1-basics/cs336_basics

# batch size: 8, 16, 32, 128,
# 学习率随 batch_size 线性改变

uv run scripts/train.py \
        --experiment_name BS_8 \
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
        --lr_max 3e-4 \
        --lr_min 3e-5 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 8 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000 


uv run scripts/train.py \
        --experiment_name BS_16 \
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
        --lr_max 3e-4 \
        --lr_min 3e-5 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 16 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000  


uv run scripts/train.py \
        --experiment_name BS_32 \
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
        --lr_max 5e-4 \
        --lr_min 5e-5 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 32 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000 

uv run scripts/train.py \
        --experiment_name BS_8 \
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
        --lr_max 2e-3 \
        --lr_min 2e-4 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 128 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000  

