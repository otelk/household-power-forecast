from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare import FEATURE_COLUMNS, has_nan, load_daily_power, matrix


def corr(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else 0.0


def main() -> None:
    out_dir = Path("outputs/eda")
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = load_daily_power()
    values = matrix(daily, FEATURE_COLUMNS)
    columns = list(zip(*values))
    out = out_dir / "feature_correlation.csv"
    with out.open("w", encoding="utf-8") as fp:
        fp.write("," + ",".join(FEATURE_COLUMNS) + "\n")
        for name, col in zip(FEATURE_COLUMNS, columns):
            fp.write(name + "," + ",".join(f"{corr(col, other):.6f}" for other in columns) + "\n")
    print(f"feature columns = {FEATURE_COLUMNS}")
    print(f"has NaN = {has_nan(daily)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
