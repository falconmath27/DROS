"""OSPG loss (P1): the paper's Algorithm 1 (Damera Venkata and Bhattacharyya, NeurIPS 2023).

A policy outputs phi_j = P(stop at j | not stopped before, history up to j), with phi_H = 1.
The stopping-time distribution on a path is exact (paper Lemma 3.1):
    psi_j = phi_j * prod_{n<j} (1 - phi_n),     psi_H = prod_{n<H} (1 - phi_n),
so the expected reward sum_j psi_j R_j is maximised directly (Theorem 4.1), with the
temporal loss of Algorithm 1 and a constant baseline (Appendix E).
"""
import torch


def log_psi(phi, eps=1e-6):
    """phi: (B, H+1) stop probabilities (the last column is ignored and treated as 1).
    Returns log psi of shape (B, H+1), computed stably in log space."""
    phi = phi.clamp(eps, 1 - eps)
    log_cont = torch.log1p(-phi[:, :-1])                       # log(1 - phi_n), n < H
    cum = torch.cumsum(log_cont, dim=1)                         # sum_{n<=j} log(1-phi_n)
    before = torch.cat([torch.zeros_like(cum[:, :1]), cum], 1)  # sum_{n<j}, j = 0..H
    lp = before.clone()
    lp[:, :-1] = lp[:, :-1] + torch.log(phi[:, :-1])
    return lp                                                   # last column: sum_{n<H}


def ospg_loss(phi, R, baseline):
    """Paper Algorithm 1: J = -mean_i sum_j stopgrad(psi_ij (r_ij - b)) log psi_ij.
    Its gradient equals minus the policy gradient of E[R_tau] (Theorem 4.1)."""
    lp = log_psi(phi)
    V = (torch.exp(lp) * (R - baseline)).detach()
    return -(V * lp).sum(dim=1).mean()
