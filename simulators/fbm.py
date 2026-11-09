"""Exact fractional Brownian motion simulator and the fBm stopping problem (P3).

Owner: P3 (stochastic simulation and rough volatility). First deliverable, end of day 2.

To do (from the project plan):
- [ ] fbm_paths: exact fBm via Cholesky or Davies-Harte
- [ ] check the empirical covariance against 0.5 (t^2H + s^2H - |t - s|^2H)
- [ ] fbm_stopping_problem: the paper's fBm table, H from 0.05 to 0.95; value ~ 0 at H = 0.5
      (reward = W^H_j itself, itm_only=False)

Shared format (see common/problem.py): every problem is a Problem(X, R) with
    X : (N, H+1, d) features observable on path i at date j (past data only)
    R : (N, H+1)    reward for stopping path i at date j, discounted to date 0
and every method has fit(X_train, R_train, X_val, R_val) and stop(X, R) -> tau.
"""
import numpy as np

from common.problem import Problem


def fbm_paths(H, n_steps=100, T=1.0, N=40_000, seed=0):
    """Return W^H on the grid t_j = j T / n_steps, shape (N, n_steps + 1), W^H_0 = 0."""
    raise NotImplementedError("P3")


def fbm_stopping_problem(H, n_steps=100, N=40_000, seed=0):
    """Problem with X = (W^H_j, j / n_steps) and R_j = W^H_j (paper Section 5)."""
    raise NotImplementedError("P3")
