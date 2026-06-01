from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.softmax import softmax
from cs336_basics.prenorm_transformer_block import (
    RMSNorm,
    SwiGluFFN,
    RotaryPositionalEmbedding,
    scaled_dot_product_attention,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerModel,
)
from cs336_basics.loss_function import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.lr_scheduling import lr_cosine_schedule
from cs336_basics.gradient_clip import gradient_clipping
from cs336_basics.dataloading import get_batch
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint

import argparse
import numpy as np

# train.py
# │
# ├── 解析命令行参数
# ├── 设置随机种子
# ├── 加载数据 (np.memmap)
# ├── 创建模型
# ├── 创建优化器
# ├── 创建学习率调度器(如果有)
# ├── 恢复checkpoint(如果有)
# │
# └── Training Loop
#       ├── train step
#       ├── optimizer step
#       ├── log train loss
#       ├── validation
#       └── save checkpoint


def get_args():

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)

    # model
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--context_len', type=int, default=1024)
    parser.add_argument('rope_theta', type=float, default=1e5)
    parser.add_argument('--rms_eps', type=float, default=1e-5)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--opt_eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    # training
    parser.add_argument('--epoches', type=int, default=5e4)
    parser.add_argument('--batch_size', type=int, default=16)

    # checkpointp
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args


if __name__ == "__main__":

    device = 'cpu'

    # 解析命令行参数
    args = get_args()

    # 加载数据 (np.memmap)
    train_tokens = np.load(
        args.train_data,
        mmap_mode='r'
    )

    val_tokens = np.load(
        args.val_data,
        mmap_mode='r'
    )

    # 创建模型
    model = TransformerModel(
        vocab_size=args.vocab_size,
        context_length=args.context_len,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        rope_theta=args.rope_theta,
        rms_eps=args.rms_eps,
    ).to(device)

    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.brta2),
        eps=args.opt_eps,
        weight_decay=args.weight_decay
    )

    # 恢复 checkpoint
    start_iter = 0

    if args.resume is not None:
        start_iter = load_checkpoint(
            args.resume,    # 待恢复model的路径
            model,
            optimizer,
        )

    # Main Training Loop