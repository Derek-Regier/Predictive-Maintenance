import torch
import torch.nn as nn

class GRUBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=bidirectional
        )
        if bidirectional:
            fc_input_dim = hidden_dim * 2
        else:
            fc_input_dim = hidden_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, output_dim)
)

    def encode(self, x):
        _, hn = self.gru(x)
        if self.bidirectional:
            out = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            out = hn[-1]
        return out

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)
