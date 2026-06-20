from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

FEATURE_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
]
TARGET_COLUMN = "global_active_power"
RAW_ZIP = Path("data/raw/individual+household+electric+power+consumption.zip")
RAW_TXT = "household_power_consumption.txt"
RAW_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


@dataclass(frozen=True)
class StandardScaler:
    mean_: list[float]
    scale_: list[float]

    @classmethod
    def fit(cls, rows: list[list[float]]) -> "StandardScaler":
        n = len(rows)
        cols = len(rows[0])
        mean = [sum(row[j] for row in rows) / n for j in range(cols)]
        scale = []
        for j in range(cols):
            var = sum((row[j] - mean[j]) ** 2 for row in rows) / n
            std = math.sqrt(var)
            scale.append(std if std else 1.0)
        return cls(mean, scale)

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        return [[(row[j] - self.mean_[j]) / self.scale_[j] for j in range(len(self.mean_))] for row in rows]


def load_daily_power(raw_zip: Path = RAW_ZIP) -> list[dict[str, float]]:
    daily: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    with ZipFile(raw_zip) as zf, zf.open(RAW_TXT) as fp:
        text = (line.decode("utf-8") for line in fp)
        reader = csv.DictReader(text, delimiter=";")
        for row in reader:
            day = datetime.strptime(row["Date"], "%d/%m/%Y").strftime("%Y-%m-%d")
            bucket = daily.setdefault(day, {col: 0.0 for col in RAW_COLUMNS})
            valid = True
            vals = {}
            for col in RAW_COLUMNS:
                if row[col] == "?":
                    valid = False
                    break
                vals[col] = float(row[col])
            if not valid:
                continue
            for col, val in vals.items():
                bucket[col] += val
            counts[day] = counts.get(day, 0) + 1

    rows: list[dict[str, float]] = []
    for day in sorted(daily):
        raw = daily[day]
        if counts.get(day, 0) == 0:
            continue
        gap = raw["Global_active_power"] * 1000.0 / 60.0 - raw["Sub_metering_1"] - raw["Sub_metering_2"] - raw["Sub_metering_3"]
        rows.append(
            {
                "date": day,
                "global_active_power": raw["Global_active_power"],
                "global_reactive_power": raw["Global_reactive_power"],
                "voltage": raw["Voltage"],
                "global_intensity": raw["Global_intensity"],
                "sub_metering_1": raw["Sub_metering_1"],
                "sub_metering_2": raw["Sub_metering_2"],
                "sub_metering_3": raw["Sub_metering_3"],
                "sub_metering_remainder": gap,
            }
        )
    if has_nan(rows):
        raise ValueError("NaN values found in daily data")
    return rows


def has_nan(rows: list[dict[str, float]]) -> bool:
    return any(math.isnan(row[col]) for row in rows for col in FEATURE_COLUMNS)


def chronological_split(rows: list[dict[str, float]], train_ratio: float = 0.7):
    split_idx = int(len(rows) * train_ratio)
    return rows[:split_idx], rows[split_idx:]


def matrix(rows: list[dict[str, float]], columns: list[str]) -> list[list[float]]:
    return [[row[col] for col in columns] for row in rows]


def scaled_daily_frames(rows: list[dict[str, float]]) -> dict[str, object]:
    train_rows, test_rows = chronological_split(rows)
    feature_scaler = StandardScaler.fit(matrix(train_rows, FEATURE_COLUMNS))
    target_scaler = StandardScaler.fit(matrix(train_rows, [TARGET_COLUMN]))
    features = feature_scaler.transform(matrix(rows, FEATURE_COLUMNS))
    target = [v[0] for v in target_scaler.transform(matrix(rows, [TARGET_COLUMN]))]
    return {
        "rows": rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "split_idx": len(train_rows),
        "features": features,
        "target": target,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
