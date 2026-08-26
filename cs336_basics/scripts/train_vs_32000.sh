#!/bin/bash

# cd ~/cs336/assignments/assignment1-basics/cs336_basics


# cd utils

# uv run dataload.py 32000

# cd ..

uv run train.py \
        --experiment_name vs_32000 \
        --train_data data/TinyStoriesV2-GPT4/vs_32000/token_ids/train/tokens.npy \
        --val_data data/TinyStoriesV2-GPT4/vs_32000/token_ids/valid/tokens.npy \
        --vocab_size 32000 \
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