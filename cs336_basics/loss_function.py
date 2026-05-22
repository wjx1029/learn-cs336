import torch

from .softmax import softmax


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        logits (Float[Tensor, "batch_size vocab_size"]): logits[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """

     # numerical stability
    max_logits = logits.max(dim=-1, keepdim=True).values

    shifed = logits - max_logits

    # logits sum exp
    logsumexp = torch.log(
        torch.sum(torch.exp(shifed), dim=-1)
    )

    target_logits = torch.gather(
        shifed,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    loss = logsumexp - target_logits

    return loss.mean()