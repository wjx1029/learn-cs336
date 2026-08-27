import torch
import os
from typing import IO, Any, BinaryIO

from cs336_basics.utils.device import try_gpu

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }

    torch.save(checkpoint, out)
    


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    
    checkpoint = torch.load(src, map_location=try_gpu())

    # 如果 checkpoint 是字典且包含 'state_dict' 键
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 去除所有键名中的 '_orig_mod.' 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_key = k[10:]  # 去掉 '_orig_mod.'（长度为10）
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # 用修改后的 state_dict 加载模型
    model.load_state_dict(new_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['iteration']