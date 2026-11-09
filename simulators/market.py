"""Market exit-timing problem: a STAND-IN price generator plus a window builder that also
works on real prices (P4 can pass a real price table to make_exit_windows).

A position is opened at date 0 of a window and must be closed by date H.
Reward for closing at date j:  R_j = S_j / S_0 - 1 - cost.
Features at date j use only prices up to date j.
"""
import numpy as np
import pandas as pd

from common.problem import Problem


def synthetic_prices(n_assets=20, n_days=3000, annual_vol=0.25, momentum=0.05,
                     garch=(0.08, 0.90), drift=0.06, seed=0):
    """STAND-IN daily prices: AR(1) returns (weak momentum) with GARCH(1,1) volatility.
    Returns a DataFrame (days x assets)."""
    rng = np.random.default_rng(seed)
    alpha, beta = garch
    var_daily = annual_vol**2 / 252
    omega = var_daily * (1 - alpha - beta)
    mu = drift / 252
    out = np.zeros((n_days, n_assets))
    for a in range(n_assets):
        h, prev, logp = var_daily, 0.0, 0.0
        for t in range(n_days):
            eps = np.sqrt(h) * rng.standard_normal()
            ret = mu + momentum * prev + eps
            out[t, a] = logp = logp + ret
            h = omega + alpha * eps**2 + beta * h
            prev = ret
    return pd.DataFrame(100 * np.exp(out), columns=[f"A{a}" for a in range(n_assets)])


def load_prices_csv(path):
    """Real data: CSV with a date column first and one column of adjusted closes per asset."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index().dropna(axis=1, how="all")


def make_exit_windows(prices, H=20, cost=0.001, vol_lookback=20, stride=1):
    """Cut every asset's price series into windows of H+1 days.

    Returns X (N, H+1, 4), R (N, H+1), start_day (N,) where the features are
        S_j/S_0 - 1                  return so far
        trailing volatility          std of the last vol_lookback daily log-returns up to j
        last daily log-return        at date j
        drawdown                     S_j / max(S_0..S_j) - 1
    """
    P = prices.to_numpy(dtype=float)
    logret = np.diff(np.log(P), axis=0, prepend=np.nan)
    vol = pd.DataFrame(logret).rolling(vol_lookback).std().to_numpy() * np.sqrt(252)
    Xs, Rs, starts = [], [], []
    n_days, n_assets = P.shape
    for a in range(n_assets):
        for s in range(vol_lookback + 1, n_days - H, stride):
            w = P[s:s + H + 1, a]
            if np.isnan(w).any() or np.isnan(vol[s:s + H + 1, a]).any():
                continue
            rel = w / w[0]
            feats = np.stack([rel - 1,
                              vol[s:s + H + 1, a],
                              logret[s:s + H + 1, a],
                              w / np.maximum.accumulate(w) - 1], axis=1)
            Xs.append(feats)
            Rs.append(rel - 1 - cost)
            starts.append(s)
    return np.array(Xs), np.array(Rs), np.array(starts)


def walk_forward_split(X, R, start_day, H, train_frac=0.6, val_frac=0.2):
    """Split by time, with a gap of H days between periods so no window spans two of them."""
    lo, hi = start_day.min(), start_day.max()
    cut1 = lo + train_frac * (hi - lo)
    cut2 = lo + (train_frac + val_frac) * (hi - lo)
    tr = start_day + H < cut1
    va = (start_day >= cut1) & (start_day + H < cut2)
    te = start_day >= cut2
    mk = lambda m: Problem(X[m], R[m], name="market_exit", itm_only=False)
    return mk(tr), mk(va), mk(te)
