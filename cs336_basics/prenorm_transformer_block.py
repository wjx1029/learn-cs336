from torch import nn
import torch
import math
import einops

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

        rms_x = torch.rsqrt(torch.mean(torch.square(x), dim = -1, keepdim=True) + self.eps)
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


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model:int, num_heads:int, rope_embedding: RotaryPositionalEmbedding=None):

        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope_embedding = rope_embedding

        self.Wqkv = Linear(d_model, d_model * 3)
        self.Wo = Linear(d_model, d_model)

    def forward(self,in_features: torch.Tensor, positions: torch.Tensor=None) -> torch.Tensor:

        seq_len = in_features.shape[-2]
        
        # project in_features to new linear space
        QKV = self.Wqkv(in_features)    # (bs, seq_len, d_model) -> (bs, seq_len, d_model * 3)
        
        # # split to Q, K, V
        # QKV = einops.rearrange(QKV, '... (n d) -> n ... d', n=3)
        # Q_proj, K_proj, V_proj = QKV.unbind(dim=0)

        # # split [num_heads] heads (bs, seq_len, d_model) -> (bs, num_heads, seq_len, d_heads)
        # Q_proj = einops.rearrange(Q_proj, '... l (h d) -> ... h l d', h=self.num_heads)
        # K_proj = einops.rearrange(K_proj, '... l (h d) -> ... h l d', h=self.num_heads)
        # V_proj = einops.rearrange(V_proj, '... l (h d) -> ... h l d', h=self.num_heads)

        # split QKV_proj to Q, K, V and split [h] heads
        QKV = einops.rearrange(
            QKV,
            "... l (three h d) -> three ... h l d",
            three=3,
            h=self.num_heads,
            d=self.d_head
        )

        Q_proj, K_proj, V_proj = QKV.unbind(dim=0)

        # rotary position embedding
        if self.rope_embedding is not None and positions is not None: 
            Q_proj = self.rope_embedding(Q_proj, positions)
            K_proj = self.rope_embedding(K_proj, positions)

        # create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)  # expand to 4 dim

        # get attention_outputs (bs, num_heads, seq_len, d_heads)
        attention_outputs = scaled_dot_product_attention(Q_proj, K_proj, V_proj, mask=causal_mask)
        # (bs, num_heads, seq_len, d_heads) -> (bs, seq_len, d_model)
        attention_outputs = einops.rearrange(attention_outputs, '... h l d -> ... l (h d)')

        return self.Wo(attention_outputs)


class TransformerBlock(nn.Module):

    def __init__(self, d_model:int, num_heads: int, d_ff: int=None, eps: float=1e-5, rope_embedding: RotaryPositionalEmbedding=None):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.rope_embedding = rope_embedding

        self.mha_layer = MultiHeadSelfAttention(d_model, num_heads, rope_embedding)

        self.swiglu_layer = SwiGluFFN(d_model, d_ff)

        self.rms_norm1 = RMSNorm(d_model, eps)

        self.rms_norm2 = RMSNorm(d_model, eps)

    def forward(self, x: torch.Tensor, tokens_position: torch.Tensor=None):

        # pre norm
        x_normed = self.rms_norm1(x)
        
        # get multi head attention
        if self.rope_embedding is not None:
            if tokens_position is not None:
                attention_outputs = self.mha_layer(x_normed, tokens_position)
            else:    
                attention_outputs = self.mha_layer(x_normed, torch.arange(x.shape[-2], device=x.device))
        else:
            attention_outputs = self.mha_layer(x_normed)
        
        # 残差链接
        mha_outputs = x + attention_outputs   

        # pre norm
        mha_outputs_normed = self.rms_norm2(mha_outputs)
        
        # SwiGlu activate
        activate_outputs = self.swiglu_layer(mha_outputs_normed)
        
        # 残差链接
        activate_outputs = activate_outputs + mha_outputs

        return activate_outputs


class TransformerModel(nn.Module):

    def __init__(self, vocab_size: int,
                       context_length: int,
                       num_layers: int,
                       d_model: int,
                       num_heads: int,
                       d_ff: int,
                       rope_theta: float,
                       rms_eps: float,
                       ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = self.num_heads
















