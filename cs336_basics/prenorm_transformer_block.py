from torch import nn
import torch
from .linear import Linear

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

        x = x1 * x2

        x = self.linear2(x)

        return x


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

        k = torch.arrange(d_k // 2, device)
        freqs = theta ** (-2 * k / d_k)

         # Shape: (max_seq_len,)
        i = torch.arrange(max_seq_len, device)

        # Shape: (max_seq_len, d_k // 2)
        angles = torch.einsum("i,k -> i k", i, freqs)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        