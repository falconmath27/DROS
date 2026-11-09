"""OSPG training (P1): the OSPG stopping method with fit() / stop(), using the loss in
ospg/loss.py and the MLP or GRU policy networks in ospg/policy.py."""
import copy

import numpy as np
import torch

from common.problem import first_true
from ospg.loss import ospg_loss
from ospg.policy import InputPrep, StepModel, predict_in_chunks


class OSPG:
    def __init__(self, kind="gru", itm_only=True, steps=20000, lr=1e-3, batch=256, seed=0,
                 eval_every=250, patience=12, min_steps=8000, init_bias=-2.0, logit_penalty=1e-3,
                 batch_norm=True, verbose=False):
        """init_bias: initial bias of the output layer, so training starts from a policy that
        mostly waits (sigmoid(-2) = 0.12). Starting near phi = 0.5 can trap the gradient in
        'stop immediately' when early rewards are large (e.g. an in-the-money put).
        Date 0 is decided separately, as for LSM and DOS: every simulated path is in the
        same state there, so the policy is trained on dates 1..H and date 0 compares the
        immediate reward with the value the trained policy achieves.
        logit_penalty: weight of mean(z^2) on the pre-sigmoid outputs z. Without it the policy
        can drive phi to ~0 at every date, where the policy gradient (proportional to phi)
        vanishes and training stalls at "never stop"; observed for the GRU on the d = 50
        max-call."""
        self.kind, self.itm_only, self.init_bias = kind, itm_only, init_bias
        self.logit_penalty = logit_penalty
        self.batch_norm = batch_norm      # MLP only: batch norm on input and hidden outputs
        self.steps, self.lr, self.batch, self.seed = steps, lr, batch, seed
        self.eval_every, self.patience, self.verbose = eval_every, patience, verbose
        self.min_steps = min_steps        # early stopping cannot end training before this

    def _phi(self, Z, return_logits=False):
        z = self.model(Z)
        phi = torch.sigmoid(z)
        if return_logits:
            return torch.cat([torch.zeros_like(phi[:, :1]), phi[:, 1:]], dim=1), z
        return torch.cat([torch.zeros_like(phi[:, :1]), phi[:, 1:]], dim=1)   # no stop at 0

    def fit(self, X, R, X_val=None, R_val=None):
        torch.manual_seed(self.seed)
        self.prep = InputPrep().fit(X, R)
        Z = self.prep(X, R)
        Rt = torch.tensor(R / self.prep.rscale, dtype=torch.float32)
        b = float(Rt.mean())                                    # constant baseline
        N = Rt.shape[0]
        self.model = StepModel(Z.shape[2], self.kind, batch_norm=self.batch_norm)
        last = self.model.net[-1] if self.kind == "mlp" else self.model.head
        torch.nn.init.constant_(last.bias, self.init_bias)
        self.stop_at_0 = False
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state, bad = -np.inf, None, 0
        for step in range(1, self.steps + 1):
            self.model.train()
            idx = torch.randint(0, N, (min(self.batch, N),))
            phi, z = self._phi(Z[idx], return_logits=True)
            loss = ospg_loss(phi, Rt[idx], b) + self.logit_penalty * (z[:, 1:] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if X_val is not None and step % self.eval_every == 0:
                tau = self.stop(X_val, R_val)
                score = R_val[np.arange(len(tau)), tau].mean()
                if self.verbose:
                    print(f"  OSPG-{self.kind} step {step}: val {score:.4f}")
                if score > best + 1e-6:
                    best, best_state, bad = score, copy.deepcopy(self.model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= self.patience and step >= self.min_steps:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        tau = self.stop(X, R)
        achieved = R[np.arange(len(tau)), tau].mean()
        self.stop_at_0 = bool(R[:, 0].mean() > achieved) and \
            (not self.itm_only or R[:, 0].mean() > 0)
        return self

    def stop(self, X, R):
        """Hard decision: stop at the first date with phi_j >= 0.5.
        Runs the network in eval mode (batch norm uses its training statistics) and in
        blocks of paths to limit memory."""
        def phi_np(Xc, Rc):
            with torch.no_grad():
                return self._phi(self.prep(Xc, Rc)).numpy()
        self.model.eval()
        try:
            phi = predict_in_chunks(phi_np, X, R)
        finally:
            self.model.train()
        mask = phi >= 0.5
        mask[:, 0] = self.stop_at_0
        if self.itm_only:
            mask &= R > 0
        return first_true(mask, R.shape[1] - 1)
