import torch
from torch import Tensor
import math

class PositionalEncoder(torch.nn.Module):
    def __init__(self, num_freqs: int):
        super(PositionalEncoder, self).__init__()
        self.num_freqs = num_freqs

    def forward(self, x: Tensor) -> Tensor:
        freq_bands = 2.0 ** torch.arange(self.num_freqs, device=x.device)  # [num_freqs]
        x = x.unsqueeze(-1) * freq_bands * math.pi  # [B*N, F, num_freqs]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [B*N, F, num_freqs*2]
        x = x.flatten(start_dim=-2)  # [B*N, F*num_freqs*2]
        return x
