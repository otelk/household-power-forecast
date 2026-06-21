from __future__ import annotations

import numpy as np


def mse(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((true - pred) ** 2))


def mae(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true - pred)))
