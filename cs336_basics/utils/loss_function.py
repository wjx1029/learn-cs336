import torch


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


def calculate_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算给定 logits 和 targets 的困惑度。

    Args:
        logits (torch.Tensor): 模型的输出，形状为 [batch_size, vocab_size]。
        targets (torch.Tensor): 真实的 token 序列，形状为 [batch_size]。

    Returns:
        float: 计算出的困惑度值。
    """
    # 1. 使用你提供的 cross_entropy 函数计算平均损失
    avg_loss = cross_entropy(logits, targets)

    # 2. 对平均损失取自然指数，得到困惑度
    perplexity = torch.exp(avg_loss)

    # 3. 将结果从 tensor 转换为 Python 的 float 类型，方便打印和比较
    return perplexity.item()
