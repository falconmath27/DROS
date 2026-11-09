"""Date-based expanding walk-forward folds with full outcome-horizon purging."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WalkForwardFold:
    number: int
    train: object
    validation: object
    test: object
    boundaries: dict


def split_at(dataset, validation_start, test_start, test_end, *, embargo_days=0, number=0):
    """Half-open periods. Train/validation outcomes must end strictly before next period.

    Shared date boundaries apply to ALL assets. Purge by scheduled deadline (not an
    early strategy exit). An optional additional calendar-day embargo is independent
    of the necessary horizon purge. Lookback history may legitimately precede test.
    """
    v, t, e = [np.datetime64(pd.Timestamp(x).date(), "D") for x in
               (validation_start, test_start, test_end)]
    if not v < t < e or embargo_days < 0:
        raise ValueError("Require validation_start < test_start < test_end and nonnegative embargo")
    start, end = dataset.metadata["entry_date"], dataset.metadata["deadline"]
    gap = np.timedelta64(embargo_days, "D")
    masks = (end < v - gap, (start >= v) & (end < t - gap), (start >= t) & (end < e))
    if any(not mask.any() for mask in masks):
        raise ValueError(f"Empty split at {v}, {t}, {e}; need more history or shorter periods")
    fold = WalkForwardFold(number, *(dataset.subset(mask) for mask in masks),
        dict(validation_start=str(v), test_start=str(t), test_end_exclusive=str(e),
             embargo_days=embargo_days))
    return fold


def walk_forward_folds(dataset, *, train_months=36, validation_months=6, test_months=6,
                       step_months=None, embargo_days=0):
    """Expanding training; disjoint test periods by default. Complete test periods only."""
    if min(train_months, validation_months, test_months) < 1:
        raise ValueError("Fold durations must be positive")
    step_months = test_months if step_months is None else step_months
    if step_months < test_months:
        raise ValueError("step_months must be >= test_months to avoid duplicate test windows")
    origin = pd.Timestamp(dataset.metadata["entry_date"].min())
    # Source coverage includes a full horizon beyond the final eligible entry.
    available = pd.Timestamp(dataset.metadata["deadline"].max()) + pd.Timedelta(days=1)
    validation = origin + pd.DateOffset(months=train_months)
    folds, number = [], 0
    while True:
        test = validation + pd.DateOffset(months=validation_months)
        end = test + pd.DateOffset(months=test_months)
        if end > available:
            break
        folds.append(split_at(dataset, validation, test, end,
                              embargo_days=embargo_days, number=number))
        validation += pd.DateOffset(months=step_months)
        number += 1
    if not folds:
        raise ValueError("Insufficient history for one complete walk-forward fold")
    return folds
