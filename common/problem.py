"""Shared data format used by every method and every problem in the project.

A stopping problem is two arrays:

    X : float array, shape (N, H+1, d)
        X[i, j, :] = features observable on path i at date j.
    R : float array, shape (N, H+1)
        R[i, j] = reward for stopping path i at date j, already discounted to date 0.

Every method implements

    fit(X_train, R_train, X_val=None, R_val=None) -> self
    stop(X, R) -> tau          # int array (N,), one stopping date per path

and may only use X[:, :j+1] and R[:, :j+1] when deciding at date j.
The score of a method on test data is evaluate(R_test, tau) = mean of R_test[i, tau_i].
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Problem:
    """A stopping problem: features X, discounted rewards R, plus metadata."""
    X: np.ndarray
    R: np.ndarray
    name: str = ""
    # True for option payoffs (never exercise with zero payoff);
    # False for trading rewards, which can be negative.
    itm_only: bool = True
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.X.ndim == 3, "X must have shape (N, H+1, d)"
        assert self.R.ndim == 2, "R must have shape (N, H+1)"
        assert self.X.shape[:2] == self.R.shape, "X and R disagree on (N, H+1)"

    @property
    def N(self):
        return self.R.shape[0]

    @property
    def H(self):
        return self.R.shape[1] - 1

    def subset(self, idx):
        return Problem(self.X[idx], self.R[idx], self.name, self.itm_only, self.info)

    def random_split(self, seed=0, test_frac=0.5, val_frac_of_train=0.2):
        """Paper protocol: random 50/50 train/test, 20% of train held out for validation."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.N)
        n_test = int(round(test_frac * self.N))
        test, train_all = perm[:n_test], perm[n_test:]
        n_val = int(round(val_frac_of_train * len(train_all)))
        val, train = train_all[:n_val], train_all[n_val:]
        return self.subset(train), self.subset(val), self.subset(test)


def evaluate(R, tau):
    """Mean reward obtained when path i stops at tau[i]."""
    tau = np.asarray(tau, dtype=int)
    assert tau.shape == (R.shape[0],)
    assert tau.min() >= 0 and tau.max() <= R.shape[1] - 1
    return float(R[np.arange(R.shape[0]), tau].mean())


def first_true(mask, H):
    """mask: (N, H+1) bool of 'stop here' decisions. Returns the first True per row,
    and H where a row has none (the last date is always a stop)."""
    mask = mask.copy()
    mask[:, H] = True
    return mask.argmax(axis=1)


class Standardizer:
    """Per-feature mean/std over all paths and dates of the training set."""

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-8
        return self

    def __call__(self, X):
        return (X - self.mean.astype(X.dtype)) / self.std.astype(X.dtype)


def check_causal(method, problem, n_paths=200, seed=0):
    """Test helper: changing data after date j must not change any decision made at or
    before j. Returns True if the method's stop() is causal on these paths."""
    rng = np.random.default_rng(seed)
    X, R = problem.X[:n_paths].copy(), problem.R[:n_paths].copy()
    tau = method.stop(X, R)
    for j in range(1, problem.H):
        X2, R2 = X.copy(), R.copy()
        X2[:, j + 1:] = X2[:, j + 1:] * (1 + rng.normal(0, 0.3, X2[:, j + 1:].shape))
        R2[:, j + 1:] = R2[:, j + 1:] * (1 + rng.normal(0, 0.3, R2[:, j + 1:].shape))
        tau2 = method.stop(X2, R2)
        early = tau <= j
        if not np.array_equal(tau[early], tau2[early]):
            return False
    return True
