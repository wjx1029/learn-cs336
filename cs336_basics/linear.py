from torch import nn
import torch

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        _std = 2 / (in_features + out_features)

        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=_std, a=_std*(-3), b=_std*3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.matmul(x, self.W.T)

        # return torch.einsum("...a, ba -> ...b", x, self.W)
        # return x @ self.W.T