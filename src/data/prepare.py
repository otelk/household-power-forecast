from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
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
PROCESSED_CSV = Path("data/processed/daily_power.csv")
RAW_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
COLUMN_MAP = {
    "Global_active_power": "global_active_power",
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3",
}
SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
]
MEAN_COLUMNS = ["voltage", "global_intensity"]


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


def _parse_float(value: str) -> float:
    return math.nan if value in {"?", ""} else float(value)


def _read_minute_rows(raw_zip: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    raw_missing = 0
    with ZipFile(raw_zip) as zf, zf.open(RAW_TXT) as fp:
        text = (line.decode("utf-8") for line in fp)
        reader = csv.DictReader(text, delimiter=";")
        for raw in reader:
            dt = datetime.strptime(f"{raw['Date']} {raw['Time']}", "%d/%m/%Y %H:%M:%S")
            row: dict[str, object] = {"datetime": dt}
            for raw_col, out_col in COLUMN_MAP.items():
                value = _parse_float(raw[raw_col])
                if math.isnan(value):
                    raw_missing += 1
                row[out_col] = value
            rows.append(row)
    rows.sort(key=lambda r: r["datetime"])
    return rows, {"raw_minute_rows": len(rows), "raw_missing_values": raw_missing}


def _interpolate_minutes(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    if not rows:
        raise ValueError("No raw minute rows found")

    by_dt = {row["datetime"]: row for row in rows}
    start = rows[0]["datetime"]
    end = rows[-1]["datetime"]
    total_minutes = int((end - start).total_seconds() // 60) + 1
    complete: list[dict[str, object]] = []
    for i in range(total_minutes):
        dt = start + timedelta(minutes=i)
        source = by_dt.get(dt, {})
        row = {"datetime": dt}
        for col in COLUMN_MAP.values():
            row[col] = source.get(col, math.nan)
        complete.append(row)

    for col in COLUMN_MAP.values():
        known = [i for i, row in enumerate(complete) if not math.isnan(row[col])]
        if not known:
            raise ValueError(f"Column {col} has no non-missing values")
        first = known[0]
        for i in range(0, first):
            complete[i][col] = complete[first][col]
        last = known[-1]
        for i in range(last + 1, len(complete)):
            complete[i][col] = complete[last][col]
        for left, right in zip(known, known[1:]):
            if right == left + 1:
                continue
            left_value = complete[left][col]
            right_value = complete[right][col]
            step = (right_value - left_value) / (right - left)
            for i in range(left + 1, right):
                complete[i][col] = left_value + step * (i - left)

    remaining_nan = sum(
        1 for row in complete for col in COLUMN_MAP.values() if math.isnan(row[col])
    )
    for row in complete:
        row["sub_metering_remainder"] = (
            row["global_active_power"] * 1000.0 / 60.0
            - row["sub_metering_1"]
            - row["sub_metering_2"]
            - row["sub_metering_3"]
        )
    return complete, {"complete_minute_rows": len(complete), "remaining_nan_values": remaining_nan}


def _aggregate_daily(minutes: list[dict[str, object]]) -> list[dict[str, float]]:
    daily: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for row in minutes:
        day = row["datetime"].strftime("%Y-%m-%d")
        bucket = daily.setdefault(day, {col: 0.0 for col in FEATURE_COLUMNS})
        for col in SUM_COLUMNS + MEAN_COLUMNS:
            bucket[col] += row[col]
        counts[day] = counts.get(day, 0) + 1

    rows: list[dict[str, float]] = []
    for day in sorted(daily):
        bucket = daily[day]
        count = counts[day]
        rows.append(
            {
                "date": day,
                "global_active_power": bucket["global_active_power"],
                "global_reactive_power": bucket["global_reactive_power"],
                "voltage": bucket["voltage"] / count,
                "global_intensity": bucket["global_intensity"] / count,
                "sub_metering_1": bucket["sub_metering_1"],
                "sub_metering_2": bucket["sub_metering_2"],
                "sub_metering_3": bucket["sub_metering_3"],
                "sub_metering_remainder": bucket["sub_metering_remainder"],
            }
        )
    return rows


def write_daily_power(rows: list[dict[str, float]], output_csv: Path = PROCESSED_CSV) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["date", *FEATURE_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def load_daily_power(raw_zip: Path = RAW_ZIP, return_metadata: bool = False):
    minute_rows, metadata = _read_minute_rows(raw_zip)
    complete_minutes, complete_metadata = _interpolate_minutes(minute_rows)
    metadata.update(complete_metadata)
    rows = _aggregate_daily(complete_minutes)
    write_daily_power(rows)
    metadata.update(
        {
            "daily_start": rows[0]["date"],
            "daily_end": rows[-1]["date"],
            "daily_continuous": is_date_continuous(rows),
            "daily_has_nan": has_nan(rows),
        }
    )
    if has_nan(rows):
        raise ValueError("NaN values found in daily data")
    return (rows, metadata) if return_metadata else rows


def has_nan(rows: list[dict[str, float]]) -> bool:
    return any(math.isnan(row[col]) for row in rows for col in FEATURE_COLUMNS)


def is_date_continuous(rows: list[dict[str, float]]) -> bool:
    dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in rows]
    return all((right - left).days == 1 for left, right in zip(dates, dates[1:]))


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
        "scaler_fit_period": (train_rows[0]["date"], train_rows[-1]["date"]),
    }
