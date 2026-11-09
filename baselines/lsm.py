"""Longstaff-Schwartz (2001): backward induction with the continuation value estimated by
least-squares regression on today's features (a Markovian method)."""
import numpy as np

from common.problem import Standardizer, first_true


class LSM:
    def __init__(self, itm_only=True, degree=2, min_points=3):
        self.itm_only = itm_only
        self.degree = degree
        self.min_points = min_points

    def _basis(self, Xj, Rj):
        """[1, x, x^2, ..., r, r^2] for standardised features x and today's reward r."""
        Z = np.hstack([self.std(Xj), Rj[:, None] / self.rscale])
        cols = [np.ones((len(Z), 1))] + [Z**k for k in range(1, self.degree + 1)]
        return np.hstack(cols)

    def fit(self, X, R, X_val=None, R_val=None):
        N, H1, _ = X.shape
        H = H1 - 1
        self.std = Standardizer().fit(X)
        self.rscale = float(np.abs(R).mean()) + 1e-8
        tau = np.full(N, H)
        self.coefs = [None] * H1
        for j in range(H - 1, 0, -1):
            cand = R[:, j] > 0 if self.itm_only else np.ones(N, bool)
            if cand.sum() < self.min_points:
                continue
            Y = R[np.arange(N), tau][cand]              # realised value of waiting
            A = self._basis(X[cand, j], R[cand, j])
            beta = np.linalg.lstsq(A, Y, rcond=None)[0]
            self.coefs[j] = beta
            C = A @ beta                                # fitted continuation value
            ex = np.zeros(N, bool)
            ex[np.where(cand)[0]] = R[cand, j] > C
            tau[ex] = j
        # Date 0: every simulated path is in the same state, so compare directly.
        self.stop_at_0 = bool(R[:, 0].mean() > R[np.arange(N), tau].mean()) and \
            (not self.itm_only or R[:, 0].mean() > 0)
        self.train_tau = tau
        return self

    def stop(self, X, R):
        N, H1, _ = X.shape
        H = H1 - 1
        mask = np.zeros((N, H1), bool)
        mask[:, 0] = self.stop_at_0
        for j in range(1, H):
            if self.coefs[j] is None:
                continue
            C = self._basis(X[:, j], R[:, j]) @ self.coefs[j]
            ok = R[:, j] > 0 if self.itm_only else np.ones(N, bool)
            mask[:, j] = ok & (R[:, j] > C)
        return first_true(mask, H)
