from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """LSTM baseline that maps multivariate history to a univariate forecast horizon."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])
