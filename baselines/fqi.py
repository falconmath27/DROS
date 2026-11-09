"""Fitted Q-iteration (paper Appendix C, Algorithm 2).

One network K(inputs up to j) estimates the continuation value at every date.
Targets are bootstrapped from the network itself:
    V_j = max(R_j, K_j) for j < H,  V_H = R_H,
and K_j is regressed on V_{j+1} (mean squared error, targets held constant).
kind='mlp' is memoryless; kind='gru' reads the history.
"""
import copy

import numpy as np
import torch

from common.problem import first_true
from ospg.policy import InputPrep, StepModel, predict_in_chunks


class FQI:
    def __init__(self, kind="mlp", itm_only=True, steps=4000, lr=1e-3, batch=512, seed=0,
                 eval_every=200, batch_norm=False):
        """batch_norm (MLP only): off by default. Unlike for OSPG, it lowered FQI-MLP on the
        American put (4.426 and 4.442 vs 4.480 and 4.462, seeds 0 and 1), so the baseline
        keeps its stronger configuration."""
        self.kind, self.itm_only, self.batch_norm = kind, itm_only, batch_norm
        self.steps, self.lr, self.batch, self.seed = steps, lr, batch, seed
        self.eval_every = eval_every

    def _K(self, Z):
        return self.model(Z)                                  # (B, H+1), scaled units

    def fit(self, X, R, X_val=None, R_val=None):
        torch.manual_seed(self.seed)
        self.prep = InputPrep().fit(X, R)
        Z = self.prep(X, R)
        Rt = torch.tensor(R / self.prep.rscale, dtype=torch.float32)
        N, H1 = Rt.shape
        self.model = StepModel(Z.shape[2], self.kind, batch_norm=self.batch_norm)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state = -np.inf, None
        for step in range(1, self.steps + 1):
            self.model.train()
            idx = torch.randint(0, N, (min(self.batch, N),))
            K = self._K(Z[idx])
            with torch.no_grad():
                V = torch.maximum(Rt[idx, :-1], K[:, :-1])
                V = torch.cat([V, Rt[idx, -1:]], dim=1)       # V_H = R_H
            loss = ((K[:, :-1] - V[:, 1:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if X_val is not None and step % self.eval_every == 0:
                tau = self.stop(X_val, R_val)
                score = R_val[np.arange(len(tau)), tau].mean()
                if score > best:
                    best, best_state = score, copy.deepcopy(self.model.state_dict())
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def stop(self, X, R):
        N, H1, _ = X.shape
        def k_np(Xc, Rc):
            with torch.no_grad():
                return self._K(self.prep(Xc, Rc)).numpy()
        self.model.eval()                       # batch norm: use training statistics
        try:
            K = predict_in_chunks(k_np, X, R) * self.prep.rscale
        finally:
            self.model.train()
        mask = R > K
        if self.itm_only:
            mask &= R > 0
        return first_true(mask, H1 - 1)
