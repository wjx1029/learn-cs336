from torch import nn
import torch
import math

from .linear import Linear
from .softmax import softmax

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    # d_model: int Hidden dimension of the model
    # eps: float = 1e-5 Epsilon value for numerical stability
    # device: torch.device | None = None Device to store the parameters on
    # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.G = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  Process an input tensor of shape(batch_size, sequence_length, d_model) and return a tensor of the same shape.

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.rsqrt(torch.mean(x * x, dim = -1, keepdim=True) + self.eps)
        result = rms_x * x * self.G

        return result.to(in_dtype)


class SwiGluFFN(nn.Module):

    def __init__(self, d_model:int, d_ff:int=None, device=None, dtype=None):
        """
        
        """
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            self.d_ff = (d_model + 32) & ~63
        else:
            self.d_ff = d_ff

        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)


        self.swish = lambda x: x / (1 + torch.exp(-x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.swish(self.linear1(x))

        x2 = self.linear3(x)


        result = self.linear2(x1 * x2)

        return result


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be input
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # shape: (d_k // 2,)
        k = torch.arange(0, d_k // 2, device=device)
        freqs = theta ** (-2 * k / d_k)

         # Shape: (max_seq_len,)
        i = torch.arange(0, max_seq_len, device=device)

        # Shape: (max_seq_len, d_k // 2)
        angles = torch.einsum("i,k->ik", i, freqs) 

        # For each pair, we need to duplicate the angle
        # We'll create cos and sin of shape (max_seq_len, d_k)
        cos_full = torch.zeros(max_seq_len, d_k, device=device)
        sin_full = torch.zeros(max_seq_len, d_k, device=device)
        
        # Fill alternating dimensions with the same angle
        cos_full[:, 0::2] = torch.cos(angles)
        cos_full[:, 1::2] = torch.cos(angles)
        sin_full[:, 0::2] = torch.sin(angles)
        sin_full[:, 1::2] = torch.sin(angles)
        
        self.register_buffer('cos', cos_full, persistent=False)
        self.register_buffer('sin', sin_full, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embeddings.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)
            
        Returns:
            Tensor with applied rotations
        """
        
        # cos_vals, sin_vals shape: (..., seq_len, d_k)
        cos_vals = self.cos[token_positions]
        sin_vals = self.sin[token_positions]

        rotate_x = self._rotate_half(x)

        result = x * cos_vals + rotate_x * sin_vals

        return result

    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape :           (..., seq_len, d_k)
        rotate_x.shape:     (..., seq_len, d_k)
        For each pair (x0, x1) -> (-x1, x0)
        """

        rotated = torch.empty_like(x)

        rotated[...,0::2] = -x[...,1::2]
        rotated[...,1::2] = x[...,0::2]

        return rotated


def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor,
                                 mask: torch.tensor=None) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    d_k = Q.shape[-1]
    
    scores = torch.einsum("...qd,...kd->...qk", Q, K)

    scores = scores / math.sqrt(d_k)

    if mask is not None:
        # tensor.masked_fill(mask, value)
        # mask: bool 张量，True 的位置会被填充
        # value: 填充的值
        scores = scores.masked_fill(~mask, float("-inf"))

    scores = softmax(scores, -1)

    attention = torch.einsum("...qk,...kd->...qd", scores, V)

    return attention