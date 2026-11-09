"""Deep Optimal Stopping (Becker, Cheridito and Jentzen, 2019).

Backward induction with one small network per date. The network at date j outputs a
soft stop probability f from today's inputs only, trained to maximise
    f * R_now + (1 - f) * R_later,
where R_later is the reward from following the (already trained) rule of later dates.
"""
import copy

import numpy as np
import torch

from common.problem import first_true
from ospg.policy import InputPrep, mlp, predict_in_chunks


class DOS:
    def __init__(self, itm_only=True, steps=1500, lr=3e-3, batch=8192, hidden=20, seed=0,
                 batch_norm=False, verbose=False):
        self.itm_only, self.batch_norm = itm_only, batch_norm
        self.steps, self.lr, self.batch, self.hidden = steps, lr, batch, hidden
        self.seed, self.verbose = seed, verbose

    def fit(self, X, R, X_val=None, R_val=None):
        torch.manual_seed(self.seed)
        N, H1, _ = X.shape
        H = H1 - 1
        self.prep = InputPrep().fit(X, R)
        Z = self.prep(X, R)                                  # (N, H+1, in_dim)
        Rt = torch.tensor(R / self.prep.rscale, dtype=torch.float32)
        tau = torch.full((N,), H, dtype=torch.long)
        self.nets = [None] * H1
        for j in range(H - 1, 0, -1):
            net = mlp(Z.shape[2], self.hidden, batch_norm=self.batch_norm) if self.nets[j + 1] is None or j == H - 1 \
                else copy.deepcopy(self.nets[j + 1])         # warm start from date j+1
            net.train()
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            now, later = Rt[:, j], Rt[torch.arange(N), tau]
            for _ in range(self.steps):
                idx = torch.randint(0, N, (min(self.batch, N),))
                f = torch.sigmoid(net(Z[idx, j]).squeeze(-1))
                loss = -(f * now[idx] + (1 - f) * later[idx]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()                                       # batch norm: training statistics
            with torch.no_grad():
                go = torch.sigmoid(net(Z[:, j]).squeeze(-1)) > 0.5
                if self.itm_only:
                    go &= now > 0
            tau[go] = j
            self.nets[j] = net
            if self.verbose and j % 10 == 0:
                print(f"  DOS date {j}: stop fraction {go.float().mean():.3f}")
        Rn = R[np.arange(N), tau.numpy()]
        self.stop_at_0 = bool(R[:, 0].mean() > Rn.mean()) and \
            (not self.itm_only or R[:, 0].mean() > 0)
        return self

    def stop(self, X, R):
        N, H1, _ = X.shape
        H = H1 - 1
        for net in self.nets:
            if net is not None:
                net.eval()

        def stop_prob(Xc, Rc):
            Z = self.prep(Xc, Rc)
            f = np.zeros(Rc.shape, dtype=np.float32)
            with torch.no_grad():
                for j in range(1, H):
                    f[:, j] = torch.sigmoid(self.nets[j](Z[:, j]).squeeze(-1)).numpy()
            return f

        f = predict_in_chunks(stop_prob, X, R)
        mask = f > 0.5
        mask[:, 0] = self.stop_at_0
        mask[:, H] = False
        if self.itm_only:
            mask[:, 1:H] &= R[:, 1:H] > 0
        return first_true(mask, H)
