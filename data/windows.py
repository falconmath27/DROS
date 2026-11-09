"""Fixed-calendar-horizon stopping problems and aligned, pickle-free metadata."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common.problem import Problem
from backtest.costs import CostModel
from data.features import build_features


@dataclass
class MarketWindows:
    problem: Problem
    metadata: dict
    feature_names: tuple
    config: dict

    def subset(self, idx):
        p = self.problem
        return MarketWindows(Problem(p.X[idx], p.R[idx], p.name, False),
            {k: v[idx] for k, v in self.metadata.items()}, self.feature_names, self.config.copy())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.problem.X, R=self.problem.R,
            feature_names=np.asarray(self.feature_names), config=np.asarray(json.dumps(self.config)),
            **self.metadata)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            reserved = {"X", "R", "feature_names", "config"}
            return cls(Problem(z["X"], z["R"], "market_exit", False),
                {k: z[k] for k in z.files if k not in reserved}, tuple(z["feature_names"].tolist()),
                json.loads(str(z["config"])))


def make_windows(prices, *, calendar_days=30, H=None, lookback=20, stride=1,
                 costs=None, market="custom", annualization=252, context=None,
                 context_lag=1, lags=(1, 2, 5), reward="linear", risk_aversion=5.0):
    """Daily-close reference-price model, with costs; not guaranteed executable close fills.

    Default deadline = entry + calendar_days; force exit on last panel session <= deadline.
    Variable session counts pad the terminal state/reward/date to calendar_days+1.
    The runner clips stopping indices to terminal_index, so padding adds no opportunity.
    H optionally selects an exact session count instead. Never shorten incomplete windows
    at the dataset end or bridge missing observations. Features 0/1 are gross return and
    running drawdown; feature 2 is remaining time. Spot return has no default discount.
    """
    if calendar_days < 1 or stride < 1 or (H is not None and H < 1):
        raise ValueError("Horizon and stride must be positive")
    if reward not in {"linear", "exponential"} or not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Use linear/exponential reward and positive risk_aversion")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Prices must have sorted unique dates")
    costs = costs or CostModel()
    features = build_features(prices, lookback, lags, annualization=annualization,
                              context=context, context_lag=context_lag)
    dates = pd.DatetimeIndex(prices.index)
    if len(dates) < 2:
        raise ValueError("Not enough price dates")
    size = H + 1 if H is not None else calendar_days + 1
    xs, rs, nets, meta = [], [], [], []
    names = ("return_since_entry", "drawdown", "time_remaining") + tuple(next(iter(features.values())).columns)
    skipped = 0
    for asset in prices:
        p = prices[asset].to_numpy(dtype=float)
        f = features[str(asset)].to_numpy(dtype=float)
        for start in range(0, len(dates), stride):
            if H is None:
                deadline = dates[start] + pd.Timedelta(days=calendar_days)
                if deadline > dates[-1]:
                    break
                end = dates.searchsorted(deadline, side="right") - 1
            else:
                end = start + H
                if end >= len(dates):
                    break
                deadline = dates[end]
            w, ff = p[start:end + 1], f[start:end + 1]
            if len(w) < 2 or not np.isfinite(w).all() or (w <= 0).any() or not np.isfinite(ff).all():
                skipped += 1
                continue
            elapsed = (dates[start:end + 1] - dates[start]).days.to_numpy()
            total = max(1, int(elapsed[-1]))
            rel = w / w[0]
            xx = np.column_stack([rel - 1, w / np.maximum.accumulate(w) - 1,
                                  1 - elapsed / total, ff])
            net = costs.net_return(rel, elapsed)
            rr = net if reward == "linear" else -np.expm1(np.clip(-risk_aversion * net, -700, 700)) / risk_aversion
            n = len(w)
            if n > size:
                raise ValueError("More than one observation per calendar day")
            xx = np.pad(xx, ((0, size - n), (0, 0)), mode="edge")
            rr = np.pad(rr, (0, size - n), mode="edge")
            dd = dates[start:end + 1].to_numpy(dtype="datetime64[D]")
            dd = np.pad(dd, (0, size - n), mode="edge")
            xs.append(xx.astype(np.float32))
            rs.append(rr)
            nets.append(np.pad(net, (0, size - n), mode="edge"))
            meta.append((str(asset), dates[start].to_datetime64(), dates[end].to_datetime64(),
                         deadline.to_datetime64(), n - 1, dd))
    if not xs:
        raise ValueError("No valid windows: check dates, lookback, context coverage and missing prices")
    metadata = dict(net_returns=np.stack(nets), asset_id=np.asarray([m[0] for m in meta]),
        market=np.full(len(meta), market),
        entry_date=np.asarray([m[1] for m in meta], dtype="datetime64[D]"),
        end_date=np.asarray([m[2] for m in meta], dtype="datetime64[D]"),
        deadline=np.asarray([m[3] for m in meta], dtype="datetime64[D]"),
        terminal_index=np.asarray([m[4] for m in meta], dtype=np.int16),
        decision_dates=np.stack([m[5] for m in meta]),
        window_id=np.asarray([f"{market}:{m[0]}:{str(np.datetime64(m[1], 'D'))}" for m in meta]))
    config = dict(calendar_days=calendar_days if H is None else None, sessions=H,
        lookback=lookback, stride=stride, annualization=annualization,
        costs=costs.to_dict(), reward=reward, risk_aversion=risk_aversion,
        execution="daily_close_reference_with_assumed_friction; not guaranteed close fills",
        context_source_observation_lag=context_lag, skipped_windows=skipped,
        padding="absorbing terminal state; tau clipped to terminal_index")
    return MarketWindows(Problem(np.stack(xs), np.stack(rs), f"{market}_exit", False), metadata, names, config)
