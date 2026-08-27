from cs336_basics.prenorm_transformer_block import TransformerModel

from cs336_basics.utils.loss_function import cross_entropy, calculate_perplexity
from cs336_basics.utils.optimizer import AdamW
from cs336_basics.utils.lr_scheduling import lr_cosine_schedule
from cs336_basics.utils.gradient_clip import gradient_clipping
from cs336_basics.utils.dataload import get_batch
from cs336_basics.utils.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.utils.device import try_gpu
from cs336_basics.utils.timer import Timer

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
import os

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

    parser.add_argument('--experiment_name', type=str, default='train')

    # data
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)

    # model
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--context_len', type=int, default=1024)
    parser.add_argument('--rope_theta', type=float, default=1e5)
    parser.add_argument('--rms_eps', type=float, default=1e-5)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--opt_eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    # learing rate schedule
    parser.add_argument("--lr_max", type=float, default=6e-4)
    parser.add_argument("--lr_min", type=float, default=6e-5)
    parser.add_argument("--warm_up", type=int, default=2000)
    parser.add_argument("--cosine_end", type=int, default=50000)

    # training
    parser.add_argument('--max_iters', type=int, default=5e4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_interval', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')

    # checkpoint
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_best_only', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=1e3)

    return parser.parse_args()


if __name__ == "__main__":

    # 解析命令行参数
    args = get_args()

    # 打印超参数
    print("=" * 100)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 100)

    # 保存超参数
    save_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'h_params.txt'), 'w') as f: 
        f.write(str(args))

    device = try_gpu()

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

    model = torch.compile(model)

    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.opt_eps,
        weight_decay=args.weight_decay
    )

    # 恢复 checkpoint
    start_iter = 1

    if args.resume is not None:
        start_iter = load_checkpoint(
            args.resume,    # 待恢复model的路径
            model,
            optimizer,
        )

    # log loss with step
    run = wandb.init(
            entity="wanjx0701-zhejiang-university",
            project="transformer-training",
            config=vars(args),
        )
    
    min_val_loss = float('inf')

    train_loss_list = []

    timer = Timer()

    # Main Training Loop
    for step in range(start_iter, args.max_iters + 1):

        model.train()

        x, y = get_batch(dataset=train_tokens,
                         batch_size=args.batch_size,
                         context_length=args.context_len,
                         device=device)
        
        # 更新学习率
        lr = lr_cosine_schedule(
            t=step,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            warm_up=args.warm_up,
            cosine_end=args.cosine_end
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        logits = model(x)

        loss = cross_entropy(
                                logits=logits.view(-1, logits.size(-1)),
                                targets=y.view(-1,)
                            )
        
        optimizer.zero_grad()

        loss.backward()

        gradient_clipping(parameters=model.parameters(),
                          max_l2_norm=1.0,
                          eps=1e-6
                          )
        
        optimizer.step()

        train_loss_list.append(loss.item())

        # 每args.val_intervals步评估一次模型
        if step != 0 and step % args.val_interval == 0:

            model.eval()

            with torch.no_grad():

                val_loss_list = []
                perplexity_list = []

                for _ in range(50):

                    x, y = get_batch(dataset=val_tokens,
                         batch_size=args.batch_size,
                         context_length=args.context_len,
                         device=device
                         )
                    
                    logits = model(x)

                    val_loss = cross_entropy(
                                             logits=logits.view(-1, logits.size(-1)),
                                             targets=y.view(-1,)
                                            )

                    perplexity = calculate_perplexity(
                                            logits=logits.view(-1, logits.size(-1)),
                                            targets=y.view(-1,)
                    )
                    
                    val_loss_list.append(val_loss.item())
                    perplexity_list.append(perplexity)

            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            avg_perplexity = sum(perplexity_list) / len(perplexity_list)
            avg_train_loss = sum(train_loss_list) / len(train_loss_list)
            train_loss_list = []    # 清除loss

            print(f"[{timer()}]  step={step}\ttrain_loss={avg_train_loss:.4f}\tval_loss={avg_val_loss:.4f}\tperplexity={avg_perplexity:.4f}")

            run.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "perplexity": avg_perplexity,
            })

            # 只保存最好的模型
            if args.save_best_only:

                if avg_val_loss < min_val_loss:

                    min_val_loss = avg_val_loss

                    checkpoint_path = os.path.join(save_dir, 'best.pth')

                    save_checkpoint(model=model,
                                    optimizer=optimizer,
                                    iteration=step,
                                    out=checkpoint_path,
                                    )


        # 每args.save_intervals步保存一次模型
        if step > 0 and step % args.save_interval == 0 and not args.save_best_only:

            checkpoint_path = os.path.join(save_dir, f'model_{step}.ckpt')

            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=step,
                            out=checkpoint_path,
                            )
        
    run.finish()

    # draw_loss(loss_list['train'], loss_list['val'])

