from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_data import build_datasets
from src.data.prepare import FEATURE_COLUMNS, load_daily_power, scaled_daily_frames
from src.models.lstm import LSTMForecaster
from src.training.metrics import mae, mse
from src.training.train_utils import set_seed


class TorchPowerDataset(Dataset):
    def __init__(self, base_ds) -> None:
        self.base_ds = base_ds

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, index: int):
        x, y = self.base_ds[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def inverse_target(values: np.ndarray, scaler) -> np.ndarray:
    return values * scaler.scale_[0] + scaler.mean_[0]


def save_predictions_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["sample", "step", "ground_truth", "prediction"])
        for sample_idx in range(y_true.shape[0]):
            for step_idx in range(y_true.shape[1]):
                writer.writerow([sample_idx, step_idx + 1, y_true[sample_idx, step_idx], y_pred[sample_idx, step_idx]])


def save_prediction_plot(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    steps = np.arange(1, y_true.shape[1] + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, y_true[0], label="Ground Truth")
    plt.plot(steps, y_pred[0], label="Prediction")
    plt.xlabel("Forecast step")
    plt.ylabel("Global active power")
    plt.title("LSTM prediction vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM baseline for household power forecasting.")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepared = scaled_daily_frames(load_daily_power())
    _, train_ds, test_ds = build_datasets(args.input_len, args.horizon, prepared)
    train_loader = DataLoader(TorchPowerDataset(train_ds), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TorchPowerDataset(test_ds), batch_size=args.batch_size, shuffle=False)

    model = LSTMForecaster(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        horizon=args.horizon,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    output_dir = ROOT / "outputs" / "lstm" / f"horizon_{args.horizon}" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, checkpoint_path)
        print(f"epoch={epoch} train_mse_scaled={train_loss:.6f}")

    y_true_scaled, y_pred_scaled = evaluate(model, test_loader, device)
    y_true = inverse_target(y_true_scaled, prepared["target_scaler"])
    y_pred = inverse_target(y_pred_scaled, prepared["target_scaler"])
    metrics = {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "input_shape": [args.batch_size, args.input_len, len(FEATURE_COLUMNS)],
        "output_shape": [args.batch_size, args.horizon],
        "test_samples": len(test_ds),
        "metrics_scale": "inverse-transformed original target scale",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_predictions_csv(output_dir / "predictions.csv", y_true, y_pred)
    save_prediction_plot(output_dir / "prediction_plot.png", y_true, y_pred)
    print(json.dumps(metrics, indent=2))
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
