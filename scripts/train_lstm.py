from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare import FEATURE_COLUMNS
from src.models.lstm import LSTMForecaster
from src.training.train_utils import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM baseline for household power forecasting.")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        args=args,
        model_name="lstm",
        model_factory=lambda: LSTMForecaster(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            horizon=args.horizon,
        ),
        output_root=ROOT / "outputs",
    )


if __name__ == "__main__":
    main()
