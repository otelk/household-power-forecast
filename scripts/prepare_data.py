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
    if not batch:
        return (0,)
    return (len(batch), len(batch[0]), len(batch[0][0])) if isinstance(batch[0][0], list) else (len(batch), len(batch[0]))


def first_batch(ds, batch_size: int):
    xs, ys = [], []
    for i in range(min(batch_size, len(ds))):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return xs, ys


def build_datasets(input_len: int, horizon: int, prepared: dict[str, object] | None = None):
    if prepared is None:
        daily = load_daily_power()
        prepared = scaled_daily_frames(daily)
    train_ds = PowerForecastDataset(prepared["features"], prepared["target"], input_len, horizon, prepared["split_idx"], "train")
    test_ds = PowerForecastDataset(prepared["features"], prepared["target"], input_len, horizon, prepared["split_idx"], "test")
    return prepared, train_ds, test_ds


def y_fully_in_test_period(ds: PowerForecastDataset, total_rows: int) -> bool:
    return all(start + ds.input_len >= ds.split_idx and start + ds.input_len + ds.horizon <= total_rows for start in ds.starts)


def print_dataset_check(prepared: dict[str, object], input_len: int, horizon: int, batch_size: int) -> None:
    _, train_ds, test_ds = build_datasets(input_len, horizon, prepared)
    train_x, train_y = first_batch(train_ds, batch_size)
    test_x, test_y = first_batch(test_ds, batch_size)
    print(f"horizon={horizon}:")
    print(f"  train samples = {len(train_ds)}")
    print(f"  test samples = {len(test_ds)}")
    print(f"  train batch X shape = {shape(train_x)}")
    print(f"  train batch y shape = {shape(train_y)}")
    print(f"  test batch X shape = {shape(test_x)}")
    print(f"  test batch y shape = {shape(test_y)}")
    print(f"  test y fully in test period = {y_fully_in_test_period(test_ds, len(prepared['rows']))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check household power forecast data windows.")
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    daily, metadata = load_daily_power(return_metadata=True)
    prepared = scaled_daily_frames(daily)
    train_rows = prepared["train_rows"]
    test_rows = prepared["test_rows"]

    print(f"raw minute rows = {metadata['raw_minute_rows']}")
    print(f"raw missing values (?/NaN) = {metadata['raw_missing_values']}")
    print(f"complete minute index rows = {metadata['complete_minute_rows']}")
    print(f"remaining NaN values after interpolate/ffill/bfill = {metadata['remaining_nan_values']}")
    print(f"daily date range = {metadata['daily_start']} to {metadata['daily_end']}")
    print(f"daily date continuous = {metadata['daily_continuous']}")
    print(f"daily has NaN = {has_nan(prepared['rows'])}")
    print(f"train date range = {train_rows[0]['date']} to {train_rows[-1]['date']}")
    print(f"test date range = {test_rows[0]['date']} to {test_rows[-1]['date']}")
    print(f"feature columns = {FEATURE_COLUMNS}")
    print(f"feature scaler fit rows = {len(train_rows)}")
    print(f"target scaler fit rows = {len(train_rows)}")
    print(f"scaler fit period = {prepared['scaler_fit_period'][0]} to {prepared['scaler_fit_period'][1]}")
    print(f"scaler fit only train period = {prepared['scaler_fit_period'] == (train_rows[0]['date'], train_rows[-1]['date'])}")
    for horizon in dict.fromkeys([args.horizon, 90, 365]):
        print_dataset_check(prepared, args.input_len, horizon, args.batch_size)


if __name__ == "__main__":
    main()
