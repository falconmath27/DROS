"""Windowing (P4): fixed calendar horizon (about 1 month of daily bars).

simulators/market.py has make_exit_windows(prices, H, cost) used by the stand-in E5:
reward R_j = S_j / S_0 - 1 - cost, features from past data only. Extend or replace it here.

Shared format (see common/problem.py): every problem is a Problem(X, R) with
    X : (N, H+1, d) features observable on path i at date j (past data only)
    R : (N, H+1)    reward for stopping path i at date j, discounted to date 0
and every method has fit(X_train, R_train, X_val, R_val) and stop(X, R) -> tau.
"""
