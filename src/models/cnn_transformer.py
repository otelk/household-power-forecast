from __future__ import annotations

import torch
from torch import nn

from src.models.transformer import PositionalEncoding


class CNNTransformerForecaster(nn.Module):
    """CNN front-end plus transformer encoder forecaster for multivariate histories."""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(input_size, d_model, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.transpose(1, 2)
        features = self.activation(self.conv(features)).transpose(1, 2)
        encoded = self.position(features)
        encoded = self.encoder(encoded)
        return self.head(encoded[:, -1])
