import torch

def softmax(x: torch.tensor, dim: int) -> torch.Tensor:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    # find the max_val in x and substract the max val for all element to avoid numerical stablility issues.
    max_val = x.max(dim=dim, keepdim=True).values

    x_trans = x - max_val

    sum_exp_x = torch.exp(x_trans).sum(dim=dim, keepdim=True)

    return torch.exp(x_trans) / sum_exp_x
