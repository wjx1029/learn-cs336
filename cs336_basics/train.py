from cs336_basics.prenorm_transformer_block import TransformerModel

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
import wandb

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

    plt.savefig()


if __name__ == "__main__":

    # 解析命令行参数
    args = get_args()

    # 打印超参数
    print("=" * 100)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 100)

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
    run = wandb.init(
            entity="wanjx0701-zhejiang-university",
            project="transformer-training",
            config=vars(args),
        )
    
    min_val_loss = float('inf')

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

        loss = cross_entropy(logits=logits.view(-1, logits.size(-1)),
                             targets=y.view(-1,)
                             )
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        gradient_clipping(parameters=model.parameters(),
                          max_l2_norm=1.0,
                          eps=1e-6
                          )
        
        optimizer.step()

        # 每args.val_intervals步评估一次模型
        if step % args.val_interval == 0:

            model.eval()

            with torch.no_grad():

                val_loss_list = []

                for _ in range(50):

                    x, y = get_batch(dataset=val_tokens,
                         batch_size=args.batch_size,
                         context_length=args.context_len,
                         device=device
                         )
                    
                    logits = model(x)

                    val_loss = cross_entropy(logits=logits.view(-1, logits.size(-1)),
                                             targets=y.view(-1,)
                                            )
                    
                    val_loss_list.append(val_loss.item())

            avg_val_loss = sum(val_loss_list) / len(val_loss_list)

            print(f"step={step} train_loss={loss.item():.4f} avg_val_loss={avg_val_loss:.4f}")

            loss_list['train'].append(loss.item())
            loss_list['val'].append(avg_val_loss)
            run.log({
                "train_loss": loss.item(),
                "val_loss": avg_val_loss,
                "step": step,
            })

            # 只保存最好的模型
            if args.save_best_only:

                if avg_val_loss < min_val_loss:

                    min_val_loss = avg_val_loss

                    checkpoint_path = f"{args.save_dir}/model_best.ckpt"

                    save_checkpoint(model=model,
                                    optimizer=optimizer,
                                    iteration=step,
                                    out=checkpoint_path,
                                    )


        # 每args.save_intervals步保存一次模型
        if step % args.save_interval == 0 and not args.save_best_only:

            checkpoint_path = f"{args.save_dir}/model_{step}.ckpt"

            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=step,
                            out=checkpoint_path,
                            )
        
    run.finish()

    # draw_loss(loss_list['train'], loss_list['val'])

