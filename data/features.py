"""Causal daily features, with source-observation lagged external context."""
import numpy as np
import pandas as pd


def build_features(prices, lookback=20, lags=(1, 2, 5), *, annualization=252,
                   context=None, context_lag=1):
    """Asset -> feature table. Models fit scaling on training only.

    Context columns are levels (index/VIX); shift >=1 source observation before
    backward alignment. Prefer local sources; daily labels lack intraday times.
    """
    if lookback < 2 or annualization <= 0 or any(lag < 1 for lag in lags):
        raise ValueError("Invalid feature parameters")
    if context is not None and context_lag < 1:
        raise ValueError("Daily external context requires at least one source-observation lag")
    result = {}
    for asset in prices:
        r = np.log(prices[asset]).diff()
        f = pd.DataFrame(index=prices.index)
        f["log_return"] = r
        f["trailing_vol"] = r.rolling(lookback, min_periods=lookback).std() * np.sqrt(annualization)
        for lag in lags:
            f[f"log_return_lag_{lag}"] = r.shift(lag)
        if context is not None:
            c = context.sort_index().shift(context_lag)
            if c.index.has_duplicates:
                raise ValueError("Context dates must be unique")
            c = c.reindex(f.index, method="ffill")
            for column in c:
                f[f"context_{column}"] = c[column]
        result[str(asset)] = f.replace([np.inf, -np.inf], np.nan)
    return result
