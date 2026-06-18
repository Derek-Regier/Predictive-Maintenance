import torch
import torch.nn as nn

class TCNBackbone(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2, output_dim=1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Build residual blocks
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal padding ensures the model doesn't leak future information
            padding = (kernel_size - 1) * dilation_size
            
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation_size),
                nn.Chopsuffix(padding) if hasattr(nn, 'Chopsuffix') else nn.Identity(), # Slice padding manually below
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            
        self.tcn = nn.Sequential(*layers)
        self.padding_to_slice = padding
        self.head = nn.Linear(num_channels[-1], output_dim)

    def encode(self, x):
        # Transpose to (batch, features, time_steps) for PyTorch Conv1d
        x = x.transpose(1, 2)
        
        # Run TCN and slice off the extra right-padding to maintain causality
        out = self.tcn(x)
        if self.padding_to_slice > 0:
            out = out[:, :, :-self.padding_to_slice]
            
        # Global max pooling over the time dimension to get an embedding per sample
        out = torch.max(out, dim=2)[0] 
        return out

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)
