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
import torch
import matplotlib.pyplot as plt

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
    parser.add_argument('--max_iters', type=int, default=5e4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_interval', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')

    # checkpointp
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=1e3)

    return parser.parse_args

def draw_loss(train_loss, val_loss=None):
    plt.figure(figsize=(8, 5))

    plt.plot(train_loss, label="train")

    if val_loss is not None:
        plt.plot(val_loss, label="validation")

    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")

    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    # 解析命令行参数
    args = get_args()

    device = args.device

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

    # log loss with step
    loss_list = {'train': [],
                 'val': []}

    # Main Training Loop
    for step in range(start_iter, args.max_iters):

        model.train()

        x, y = get_batch(dataset=train_tokens,
                         batch_size=args.batch_size,
                         context_length=args.context_len,
                         device=device)
        
        logits = model(x)

        loss = cross_entropy(logits=logits.view(-1, logits.size(-1)),
                             targets=y.view(-1,)
                             )
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.backward()

        gradient_clipping(parameters=model.parameters(),
                          max_l2_norm=1.0,
                          eps=1e-6
                          )
        
        optimizer.step()

        # 每args.val_intervals步评估一次模型
        if step % args.val_intervals == 0:

            model.eval()

            with torch.no_grad():

                val_loss = []

                for _ in range():

                    x, y = get_batch(dataset=val_tokens,
                         batch_size=args.batch_size,
                         context_length=args.context_len,
                         device=device
                         )
                    
                    logits = model(x)

                    val_loss = cross_entropy(logits=logits.view(-1, logits.size(-1)),
                                             targets=y.view(-1,)
                                            )
                    
                    val_loss.append(val_loss.item())

            avg_val_loss = sum(val_loss) / len(val_loss)

            print(f"step={step} train_loss={loss.item():.4f} avg_val_loss={avg_val_loss:.4f}")

            loss_list['train'].append(loss.item())
            loss_list['val'].append(avg_val_loss)

        # 每args.save_intervals步保存一次模型
        if step % args.save_intervals == 0:

            checkpoint_path = f"{args.save_dir}/model_{step}.ckpt"

            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=step,
                            out=checkpoint_path,
                            )
