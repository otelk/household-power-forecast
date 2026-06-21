from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare import FEATURE_COLUMNS
from src.models.cnn_transformer import CNNTransformerForecaster
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster
from src.training.train_utils import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a household power forecasting model.")
    parser.add_argument("--model", choices=("lstm", "transformer", "cnn_transformer"), required=True)
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer embedding size.")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kernel-size", type=int, default=3, help="CNN kernel size for cnn_transformer.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    if args.model == "lstm":
        return LSTMForecaster(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            horizon=args.horizon,
        )
    if args.model == "transformer":
        return TransformerForecaster(
            input_size=len(FEATURE_COLUMNS),
            horizon=args.horizon,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
    if args.model == "cnn_transformer":
        return CNNTransformerForecaster(
            input_size=len(FEATURE_COLUMNS),
            horizon=args.horizon,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            kernel_size=args.kernel_size,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        args=args,
        model_name=args.model,
        model_factory=lambda: build_model(args),
        output_root=ROOT / "outputs",
    )


if __name__ == "__main__":
    main()
