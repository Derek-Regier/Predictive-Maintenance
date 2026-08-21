import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        positional_encode = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encode[:, 0::2] = torch.sin(position * div_term)
        positional_encode[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', positional_encode.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class TransformerBackbone(nn.Module):
    def __init__(self, input_dim, d_model, nhead, dropout, num_layers, dim_feedforward, output_dim=1):
        super().__init__()
        # Project raw input features to the Transformer's internal d_model dimension
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim)
        )

    def encode(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        # Pull the final time step representation out as the sequence summary
        return out[:, -1, :] 

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)
