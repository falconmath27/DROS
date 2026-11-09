"""Policy networks (P1): the MLP (2 x 20, batch norm) and GRU (1 x 20) stopping policies,
and the input preparation shared by OSPG, DOS and FQI."""
import numpy as np
import torch
import torch.nn as nn

from common.problem import Standardizer

torch.set_num_threads(max(1, torch.get_num_threads()))


class MLP(nn.Sequential):
    """Feed-forward network applied to the last axis of its input, so it accepts (B, d) or
    (B, T, d). With batch_norm=True, batch normalisation is applied to the input and to the
    output of every hidden activation, as in the paper (Appendix D.1) and in Becker et al.
    Statistics are pooled over paths and dates. Call .eval() before making decisions so
    the running statistics from training are used (then every decision is causal)."""

    def forward(self, x):
        lead = x.shape[:-1]
        y = super().forward(x.reshape(-1, x.shape[-1]))
        return y.reshape(*lead, y.shape[-1])


def mlp(in_dim, hidden=20, layers=2, out=1, batch_norm=True):
    mods, d = [], in_dim
    if batch_norm:
        mods.append(nn.BatchNorm1d(in_dim))
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        if batch_norm:
            mods.append(nn.BatchNorm1d(hidden))
        d = hidden
    mods.append(nn.Linear(d, out))
    return MLP(*mods)


class StepModel(nn.Module):
    """Maps a sequence of inputs (B, T, d) to one output per step (B, T).
    kind='mlp': each step uses only its own inputs (memoryless, weights shared over time),
                two hidden layers of 20 units with batch normalisation.
    kind='gru': a single-layer GRU reads the history up to each step."""

    def __init__(self, in_dim, kind="gru", hidden=20, batch_norm=True):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.net = mlp(in_dim, hidden, layers=2, batch_norm=batch_norm)
        elif kind == "gru":
            self.gru = nn.GRU(in_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
        else:
            raise ValueError(kind)

    def forward(self, Z):
        if self.kind == "mlp":
            return self.net(Z).squeeze(-1)
        h, _ = self.gru(Z)
        return self.head(h).squeeze(-1)


class InputPrep:
    """Builds network inputs at every date: standardised features, the current reward
    (scaled), and the time fraction j/H. All three are known at date j.
    Works in float32 to keep memory down on large problems."""

    def fit(self, X, R):
        self.std = Standardizer().fit(X)
        self.rscale = float(np.abs(R).mean()) + 1e-8
        return self

    def __call__(self, X, R):
        N, T, _ = X.shape
        tfrac = np.broadcast_to(np.linspace(0, 1, T, dtype=np.float32)[None, :, None], (N, T, 1))
        Z = np.concatenate([self.std(X).astype(np.float32, copy=False),
                            (R / self.rscale).astype(np.float32)[:, :, None], tfrac], axis=2)
        return torch.from_numpy(Z)

    @property
    def dim_extra(self):
        return 2


def predict_in_chunks(fn, X, R, chunk=8192):
    """Apply fn(X_chunk, R_chunk) -> array (n, ...) to blocks of paths and stack the results,
    so large test sets do not need all network inputs in memory at once."""
    return np.concatenate([fn(X[a:a + chunk], R[a:a + chunk]) for a in range(0, len(R), chunk)])


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
