import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    """Slices off right-side padding to enforce causality after left-padded conv."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation   # left-only padding amount
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 1×1 conv residual when channel dimensions change, identity otherwise
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(self.net(x) + self.residual(x))


class TCNBackbone(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2, output_dim=1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Build residual blocks sequentially
        for i in range(num_levels):
            dilation_size = 2 ** i
            if i == 0 :
                in_channels = input_dim 
            else:
                in_channels = num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation_size, dropout)
            )
            
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], output_dim)

    def encode(self, x):
        # Transpose to (batch, features, time_steps) for PyTorch Conv1d
        x = x.transpose(1, 2)          # (batch, features, time)
        out = self.tcn(x)               # (batch, channels, time)
        
        # Extract the last timestep since causality guarantees it holds the full sequence context
        return out[:, :, -1]            # (batch, channels)

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)