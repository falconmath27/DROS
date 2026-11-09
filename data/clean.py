"""Conservative cleaning with audit records; no interpolation or future-price filling."""
import numpy as np
import pandas as pd
from data.loaders import STABLECOINS


def clean_prices(prices, max_gap_days=5, outlier_z=10.0, *, min_observations=60,
                 fill_short_gaps=False, calendar="equity"):
    """Return (clean table, log). Default leaves gaps; flag jumps without deleting crashes.

    Equity sessions are observed panel dates; crypto uses every UTC day. Optional
    forward fill is bounded by elapsed missing observations, not future gap length.
    """
    if max_gap_days < 0 or min_observations < 1 or outlier_z <= 0:
        raise ValueError("Invalid cleaning parameters")
    if calendar not in {"equity", "crypto"}:
        raise ValueError("calendar must be equity or crypto")
    frame, attrs, log = prices.copy(), dict(prices.attrs), []
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    duplicates = int(frame.index.duplicated(keep="last").sum())
    if duplicates:
        log.append(dict(action="duplicate_dates_keep_last", count=duplicates))
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate asset columns")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.empty:
        raise ValueError("No prices supplied")
    if calendar == "crypto":
        frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="D"))
    for symbol in list(frame):
        base = str(symbol).replace("/", "").removesuffix("USDT").removesuffix("-USD")
        if str(symbol).upper() in STABLECOINS or base.upper() in STABLECOINS:
            frame = frame.drop(columns=symbol)
            log.append(dict(asset=str(symbol), action="drop_stablecoin"))
            continue
        s = frame[symbol].copy()
        invalid = ~np.isfinite(s.to_numpy()) | (s.to_numpy() <= 0)
        count = int((s.notna().to_numpy() & invalid).sum())
        s.iloc[np.flatnonzero(invalid)] = np.nan
        if count:
            log.append(dict(asset=str(symbol), action="invalid_price_to_missing", count=count))
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None or s.count() < min_observations:
            frame = frame.drop(columns=symbol)
            log.append(dict(asset=str(symbol), action="drop_insufficient_history", count=int(s.count())))
            continue
        interior = s.loc[first:last]
        if interior.isna().any():
            log.append(dict(asset=str(symbol), action="interior_missing", count=int(interior.isna().sum())))
        if fill_short_gaps and max_gap_days:
            filled = interior.ffill(limit=max_gap_days)
            n = int((interior.isna() & filled.notna()).sum())
            s.loc[first:last] = filled
            log.append(dict(asset=str(symbol), action="causal_forward_fill", count=n))
        returns = np.log(s).diff()
        mean = returns.shift(1).rolling(60, min_periods=20).mean()
        std = returns.shift(1).rolling(60, min_periods=20).std()
        flagged = ((returns - mean).abs() > outlier_z * std) & (std > 0)
        for date in flagged.index[flagged]:
            log.append(dict(asset=str(symbol), action="flag_jump_retained", date=str(date.date()),
                            log_return=float(returns.loc[date])))
        log.append(dict(asset=str(symbol), action="coverage", first=str(first.date()),
            last=str(last.date()), observations=int(s.count()),
            note="No listing/delisting extrapolation; terminal delisting proceeds unknown."))
        frame[symbol] = s
    if frame.shape[1] == 0:
        raise ValueError("No assets survived cleaning")
    frame.index.name = "date"
    frame.attrs = attrs
    return frame, log
