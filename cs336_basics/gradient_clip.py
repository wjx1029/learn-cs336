from collections.abc import Iterable
import torch
import math


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm after each backward pass before taking an optimizer step.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    
    params = [p for p in parameters if p.grad is not None]

    total_norm = torch.sqrt(
        sum(
            torch.sum(p.grad * p.grad)
            for p in params
        )
    )

    if total_norm >= max_l2_norm:
        
        scale = max_l2_norm / (total_norm + eps)

        for p in params:
            p.grad.mul_(scale)
