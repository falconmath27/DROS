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
    """Return W^H on the grid t_j = jT/n_steps, shape (N, n_steps + 1)."""

    if not 0 < H < 1:
        raise ValueError("H must be between 0 and 1")

    if n_steps < 1 or N < 1 or T <= 0:
        raise ValueError("Invalid simulation parameters")

    rng = np.random.default_rng(seed)

    # Time grid, excluding t=0
    t = np.linspace(0, T, n_steps + 1)[1:]

    # Theoretical covariance matrix
    C = 0.5 * (
        t[:, None]**(2*H)
        + t[None, :]**(2*H)
        - np.abs(t[:, None] - t[None, :])**(2*H)
    )

        # Cholesky decomposition
    L = np.linalg.cholesky(C)

    # Generate independent standard normal variables
    Z = rng.standard_normal((N, n_steps))

    # Transform into correlated Gaussian variables
    W = Z @ L.T

    # Add initial value W(0) = 0
    W = np.column_stack((np.zeros(N), W))

    return W


def fbm_stopping_problem(H, n_steps=100, N=40_000, seed=0):
    """Problem with X = (W^H_j, j/n_steps) and R_j = W^H_j."""

    # Generate fBm paths
    W = fbm_paths(H, n_steps, N=N, seed=seed)

    # Normalised time feature: 0, 1/n_steps, ..., 1
    times = np.linspace(0, 1, n_steps + 1)

    # Repeat time feature across all paths
    time_feature = np.broadcast_to(times, (N, n_steps + 1))

    # Construct features: [fBm value, normalised time]
    X = np.stack([W, time_feature], axis=-1)

    # Reward is simply the fBm value
    R = W.copy()

    return Problem(
        X=X,
        R=R,
        name=f"fBm stopping (H={H})",
        itm_only=False
    )
