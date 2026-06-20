from __future__ import annotations

from src.data.prepare import FEATURE_COLUMNS


class PowerForecastDataset:
    """Sliding-window dataset with multivariate history and univariate target."""

    def __init__(self, features, target, input_len: int = 90, horizon: int = 90, split_idx: int | None = None, split: str = "train") -> None:
        if not features or len(features[0]) != len(FEATURE_COLUMNS):
            raise ValueError(f"features must have shape [time, {len(FEATURE_COLUMNS)}]")
        if len(features) != len(target):
            raise ValueError("features and target must have matching time length")
        if split_idx is None:
            split_idx = int(len(target) * 0.7)
        self.features = features
        self.target = target
        self.input_len = input_len
        self.horizon = horizon
        self.split_idx = split_idx
        if split == "train":
            self.starts = list(range(0, split_idx - input_len - horizon + 1))
        elif split == "test":
            self.starts = list(range(split_idx - input_len, len(target) - input_len - horizon + 1))
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = self.starts[index]
        x_end = start + self.input_len
        y_end = x_end + self.horizon
        return self.features[start:x_end], self.target[x_end:y_end]
