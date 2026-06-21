from __future__ import annotations

import subprocess
import sys


def main() -> None:
    for horizon in (90, 365):
        subprocess.run(
            [sys.executable, "scripts/train_lstm.py", "--horizon", str(horizon)],
            check=True,
        )


if __name__ == "__main__":
    main()
