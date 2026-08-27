#!/bin/bash

# 新增实验
# 1. 补充batchsize实验 batch_size=80

uv run train.py \
        --experiment_name BS_80 \
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
        --lr_max 1.25e-3 \
        --lr_min 1.25e-4 \
        --warm_up 500 \
        --cosine_end 5000 \
        \
        --max_iters 5000 \
        --batch_size 80 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000  

# 补充实验2: 在base的基础上,加大训练轮次为10000
uv run train.py \
        --experiment_name base_iter_10000 \
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
        --lr_max 1e-3 \
        --lr_min 1e-4 \
        --warm_up 500 \
        --cosine_end 8000 \
        \
        --max_iters 10000 \
        --batch_size 64 \
        --val_interval 100 \
        --device cpu \
        \
        --save_dir runs \
        --save_best_only \
        --save_interval 1000 