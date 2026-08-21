import torch
import torch.nn as nn

class LSTMBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # batch_first=True expects shape: (batch_size, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Account for bidirectional output doubling the hidden dimension size
        fc_input_dim = hidden_dim
        if bidirectional:
            fc_input_dim *= 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, output_dim)
)

    def encode(self, x):
        """Returns (batch, hidden_dim) — used by the stacking layer."""
        _, (hn, _) = self.lstm(x)
        
        if self.bidirectional:
            # Concat the forward and backward hidden states of the very last layer
            out = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            # Extract the hidden state of the very last layer
            out = hn[-1]
        return out

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)
