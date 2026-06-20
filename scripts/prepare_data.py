from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import PowerForecastDataset
from src.data.prepare import FEATURE_COLUMNS, has_nan, load_daily_power, scaled_daily_frames


def shape(batch):
    return (len(batch), len(batch[0]), len(batch[0][0])) if isinstance(batch[0][0], list) else (len(batch), len(batch[0]))


def first_batch(ds, batch_size: int):
    xs, ys = [], []
    for i in range(min(batch_size, len(ds))):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return xs, ys


def build_datasets(input_len: int, horizon: int):
    daily = load_daily_power()
    prepared = scaled_daily_frames(daily)
    train_ds = PowerForecastDataset(prepared["features"], prepared["target"], input_len, horizon, prepared["split_idx"], "train")
    test_ds = PowerForecastDataset(prepared["features"], prepared["target"], input_len, horizon, prepared["split_idx"], "test")
    return prepared, train_ds, test_ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check household power forecast data windows.")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    prepared, train_ds, test_ds = build_datasets(args.input_len, args.horizon)
    train_x, train_y = first_batch(train_ds, args.batch_size)
    test_x, test_y = first_batch(test_ds, args.batch_size)

    print(f"feature columns = {FEATURE_COLUMNS}")
    print(f"feature scaler fit rows = {len(prepared['train_rows'])}")
    print(f"target scaler fit rows = {len(prepared['train_rows'])}")
    print(f"has NaN = {has_nan(prepared['rows'])}")
    print(f"horizon={args.horizon}:")
    print(f"train samples = {len(train_ds)}")
    print(f"test samples = {len(test_ds)}")
    print(f"train batch X shape = {shape(train_x)}")
    print(f"train batch y shape = {shape(train_y)}")
    print(f"test batch X shape = {shape(test_x)}")
    print(f"test batch y shape = {shape(test_y)}")
    first_test_y_start = test_ds.starts[0] + args.input_len
    print(f"first test y index = {first_test_y_start}; split index = {prepared['split_idx']}")
    print(f"test y fully in test period = {first_test_y_start >= prepared['split_idx']}")


if __name__ == "__main__":
    main()
