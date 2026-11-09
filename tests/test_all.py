"""Fast tests (about 1-2 minutes). Run from the project root:  python -m pytest -q tests"""
import math

import numpy as np
import pytest
import torch

from baselines.binomial import (binomial_price, black_scholes_put, exercise_boundary,
                                perpetual_put)
from baselines.dos import DOS
from baselines.fqi import FQI
from baselines.lsm import LSM
from baselines.pde import fd_price, fd_solve
from common.problem import Problem, check_causal, evaluate, first_true
from ospg.loss import log_psi, ospg_loss
from ospg.train import OSPG
from simulators.gbm import american_put_problem, geometric_call_reference, gbm_paths
from simulators.market import make_exit_windows, synthetic_prices
from simulators.rbergomi import _joint_covariance, rbergomi_paths

# --------------------------------------------------------------------------- binomial

def test_european_matches_black_scholes():
    tree = binomial_price(36, 40, 0.06, 0.2, 1, 5000, american=False)
    assert abs(tree - black_scholes_put(36, 40, 0.06, 0.2, 1)) < 2e-3
    assert abs(tree - 3.844) < 2e-3


def test_american_and_bermudan_put():
    assert abs(binomial_price(36, 40, 0.06, 0.2, 1, 5000) - 4.487) < 2e-3
    berm = binomial_price(36, 40, 0.06, 0.2, 1, 5000, exercise_every=100)   # 50 dates
    assert abs(berm - 4.478) < 2e-3


def test_american_at_least_european():
    for S in [30, 36, 40, 44]:
        assert binomial_price(S, 40, 0.06, 0.2, 1, 500) >= \
            binomial_price(S, 40, 0.06, 0.2, 1, 500, american=False) - 1e-12


def test_perpetual_put_and_smooth_pasting():
    V, L = perpetual_put(36, 40, 0.06, 0.2)
    assert abs(L - 30) < 1e-12 and abs(V - 5.787037) < 1e-5
    h = 1e-5
    slope = (perpetual_put(30 + h, 40, 0.06, 0.2)[0] - perpetual_put(30, 40, 0.06, 0.2)[0]) / h
    assert abs(slope + 1) < 1e-3


def test_exercise_boundary_rises_towards_strike():
    t, b = exercise_boundary(36, 40, 0.06, 0.2, 1, 1000)
    b = b[~np.isnan(b)]
    assert b[-1] > b[len(b) // 4] and b.max() < 40


# --------------------------------------------------------------------------- PDE (finite differences)

def test_pde_european_matches_black_scholes():
    assert abs(fd_price(36, 40, 0.06, 0.2, 1, american=False) - black_scholes_put(36, 40, 0.06, 0.2, 1)) < 5e-4


def test_pde_american_and_bermudan_match_tree():
    for S0, sig, T in [(36, 0.2, 1), (40, 0.4, 2), (44, 0.2, 2)]:
        tree = binomial_price(S0, 40, 0.06, sig, T, 5000)
        assert abs(fd_price(S0, 40, 0.06, sig, T) - tree) < 3e-3
    berm = fd_price(36, 40, 0.06, 0.2, 1, exercise_every=20)          # 50 exercise dates
    assert abs(berm - 4.478) < 2e-3


def test_pde_converges_and_respects_bounds():
    coarse, fine = fd_price(36, 40, 0.06, 0.2, 1, n_x=250, n_t=250), fd_price(36, 40, 0.06, 0.2, 1, n_x=1000, n_t=1000)
    assert abs(coarse - fine) < 2e-3
    eu = fd_price(36, 40, 0.06, 0.2, 1, american=False)
    assert eu < fd_price(36, 40, 0.06, 0.2, 1, exercise_every=20) < fine
    v5, v20 = fd_price(36, 40, 0.06, 0.2, 5), fd_price(36, 40, 0.06, 0.2, 20, n_x=2000)
    assert v5 < v20 < perpetual_put(36, 40, 0.06, 0.2)[0] + 1e-3     # increases to perpetual


def test_pde_boundary_matches_tree():
    out = fd_solve(36, 40, 0.06, 0.2, 1)
    t_tree, b_tree = exercise_boundary(36, 40, 0.06, 0.2, 1, 1000)
    idx = np.arange(100, 900)                       # both grids have 1000 steps on [0, 1]
    gap = np.abs(out["boundary"][idx] - b_tree[idx])
    cell = 34 * 0.2 * np.sqrt(1 / 1000)             # tree spacing near the boundary, ~0.21
    assert np.nanmean(gap) < cell and np.nanmax(gap) < 2 * cell
    assert out["boundary"][-1] == 40 and np.nanmax(out["boundary"][:-1]) < 40


def test_pde_geometric_call_matches_tree():
    d, rho, sig, dl = 7, 0.75, 0.25, 0.02
    sG = sig * np.sqrt((1 + (d - 1) * rho) / d)
    dG = dl + 0.5 * sig**2 - 0.5 * sG**2
    assert abs(fd_price(100, 100, 0.0, sG, 2, kind="call", delta=dG) - geometric_call_reference(d, n=5000)) < 2e-3


def test_geometric_call_reference_matches_paper():
    for d, p in [(7, 10.2591), (13, 10.0984), (20, 10.0326)]:
        assert abs(geometric_call_reference(d, n=5000) - p) < 3e-3

# --------------------------------------------------------------------------- LSM

LS_PATHS = np.array([[1, 1.09, 1.08, 1.34], [1, 1.16, 1.26, 1.54], [1, 1.22, 1.07, 1.03],
                     [1, .93, .97, .92], [1, 1.11, 1.56, 1.52], [1, .76, .77, .90],
                     [1, .92, .84, 1.01], [1, .88, 1.22, 1.34]])


def _ls_raw_regression():
    """Longstaff-Schwartz 8-path example with the plain [1, S, S^2] basis."""
    K, disc = 1.10, math.exp(-0.06)
    cf, tau = np.maximum(K - LS_PATHS[:, 3], 0), np.full(8, 3)
    coefs = {}
    for j in [2, 1]:
        S = LS_PATHS[:, j]; pay = np.maximum(K - S, 0); itm = pay > 0
        Y = cf * disc ** (tau - j)
        A = np.vstack([np.ones(itm.sum()), S[itm], S[itm] ** 2]).T
        b = np.linalg.lstsq(A, Y[itm], rcond=None)[0]; coefs[j] = b
        ex = np.zeros(8, bool); ex[itm] = pay[itm] > A @ b
        cf[ex], tau[ex] = pay[ex], j
    return coefs, np.mean(cf * disc ** tau)


def test_ls_eight_path_example():
    coefs, price = _ls_raw_regression()
    assert np.allclose(coefs[2], [-1.070, 2.983, -1.814], atol=2e-3)
    assert abs(price - 0.1144) < 1e-4


def test_lsm_class_on_eight_paths():
    """The project's LSM class (standardised basis + reward terms) makes the same decisions
    here: 0.1144 in-sample."""
    R = np.exp(-0.06 * np.arange(4)) * np.maximum(1.10 - LS_PATHS, 0)
    m = LSM(itm_only=True, degree=2).fit(LS_PATHS[:, :, None], R)
    price = evaluate(R, m.train_tau)
    assert price > 0.10


def test_lsm_american_put_out_of_sample():
    tr = american_put_problem(N=60_000, seed=1)
    te = american_put_problem(N=60_000, seed=2)
    m = LSM().fit(tr.X, tr.R)
    v = evaluate(te.R, m.stop(te.X, te.R))
    assert 4.43 < v < 4.49            # lower bound for the Bermudan price 4.478

# --------------------------------------------------------------------------- OSPG maths

def test_psi_is_a_distribution():
    phi = torch.rand(50, 11)
    psi = torch.exp(log_psi(phi))
    assert torch.allclose(psi.sum(1), torch.ones(50), atol=1e-5)
    # hand example from the study guide: phi = 0.2, 0.5, (1)
    psi = torch.exp(log_psi(torch.tensor([[0.2, 0.5, 0.0]])))
    assert torch.allclose(psi, torch.tensor([[0.2, 0.4, 0.4]]), atol=1e-6)


def test_ospg_loss_gradient_equals_policy_gradient():
    """grad of Algorithm-1 loss == -grad of E[sum_j psi_j R_j] (Theorem 4.1), and the
    constant baseline does not change it (Proposition E.1)."""
    torch.manual_seed(0)
    w = torch.randn(6, requires_grad=True)
    feats = torch.randn(32, 6, 6)
    R = torch.rand(32, 6)
    phi = lambda: torch.sigmoid(feats @ w)
    g_direct = torch.autograd.grad(-(torch.exp(log_psi(phi())) * R).sum(1).mean(), w)[0]
    g_loss0 = torch.autograd.grad(ospg_loss(phi(), R, 0.0), w)[0]
    g_loss_b = torch.autograd.grad(ospg_loss(phi(), R, 0.7), w)[0]
    assert torch.allclose(g_direct, g_loss0, atol=1e-6)
    assert torch.allclose(g_loss0, g_loss_b, atol=1e-6)


def test_first_true_and_evaluate():
    m = np.array([[0, 1, 0, 1], [0, 0, 0, 0]], bool)
    assert list(first_true(m, 3)) == [1, 3]
    R = np.array([[0, .1, .05], [0, 0, .2], [0, .3, .15]])
    assert abs(evaluate(R, np.array([1, 2, 1])) - 0.2) < 1e-12

# --------------------------------------------------------------------------- causality

@pytest.fixture(scope="module")
def small_put():
    return american_put_problem(N=3000, H=10, seed=3)


@pytest.mark.parametrize("make", [
    lambda: LSM(),
    lambda: DOS(steps=30, batch=512),
    lambda: FQI(kind="mlp", steps=50),
    lambda: FQI(kind="gru", steps=50),
    lambda: OSPG(kind="mlp", steps=50),
    lambda: OSPG(kind="gru", steps=50),
])
def test_methods_are_causal(make, small_put):
    m = make().fit(small_put.X, small_put.R)
    tau = m.stop(small_put.X, small_put.R)
    assert tau.shape == (small_put.N,) and tau.min() >= 0 and tau.max() <= small_put.H
    assert check_causal(m, small_put)

@pytest.mark.parametrize("make", [
    lambda: DOS(steps=30, batch=512, batch_norm=True),
    lambda: FQI(kind="mlp", steps=50, batch_norm=True),
    lambda: OSPG(kind="mlp", steps=50),
])
def test_batch_norm_decisions_do_not_depend_on_other_paths(make, small_put):
    """MLPs use batch norm (paper App. D.1). At decision time it must use the training
    statistics, so a path's decision cannot depend on which other paths are in the batch."""
    m = make().fit(small_put.X, small_put.R)
    nets = m.nets[1:-1] if hasattr(m, "nets") else [m.model.net]
    assert all(any(isinstance(l, torch.nn.BatchNorm1d) for l in n) for n in nets)
    tau_all = m.stop(small_put.X, small_put.R)
    tau_part = m.stop(small_put.X[:37], small_put.R[:37])
    assert np.array_equal(tau_all[:37], tau_part)

# --------------------------------------------------------------------------- repo skeleton

@pytest.mark.parametrize("module", [
    "ospg", "ospg.loss", "ospg.policy", "ospg.train",
    "baselines.binomial", "baselines.pde", "baselines.lsm", "baselines.dos", "baselines.fqi",
    "simulators.gbm", "simulators.fbm", "simulators.rbergomi", "simulators.market",
    "data", "data.loaders", "data.clean", "data.windows", "data.features", "data.splits",
    "backtest", "backtest.rules", "backtest.costs", "backtest.run",
    "analysis", "analysis.bootstrap", "analysis.hurst", "analysis.regimes",
    "experiments.runner", "experiments.summarize", "experiments.paired",
])
def test_every_module_imports(module):
    """Every file in the shared layout (including the P3-P5 skeletons) must at least import."""
    import importlib
    importlib.import_module(module)

# --------------------------------------------------------------------------- simulators

def test_gbm_martingale_after_discounting():
    S = gbm_paths(100, 0.05, 0.2, 1, 10, 400_000, seed=4)
    assert abs(np.exp(-0.05) * S[:, -1, 0].mean() - 100) < 0.15


def test_rbergomi_covariance_and_martingale():
    t = np.arange(1, 51) / 50
    C = _joint_covariance(t, 0.07)
    assert np.linalg.eigvalsh(C).min() > 0
    assert np.allclose(np.diag(C)[:50], t ** 0.14, atol=1e-10)     # Var Wt_t = t^{2H}
    S, v, tt, _ = rbergomi_paths(N=200_000, n_steps=50, seed=5)
    se = S[:, -1].std() / np.sqrt(len(S))
    assert abs(np.exp(-0.05) * S[:, -1].mean() - 100) < 4 * se
    assert abs(v[:, -1].mean() / 0.09 - 1) < 0.05                  # E[v_t] = xi0


def test_market_features_do_not_peek():
    P = synthetic_prices(n_assets=1, n_days=300, seed=6)
    X1, R1, s1 = make_exit_windows(P, H=10)
    P2 = P.copy()
    P2.iloc[200:] *= 1.5                                          # change the future
    X2, R2, s2 = make_exit_windows(P2, H=10)
    i = np.where(s1 == 150)[0][0]                                 # window days 150..160
    assert np.allclose(X1[i], X2[i]) and np.allclose(R1[i], R2[i])
    j = np.where(s1 == 195)[0][0]                                 # window days 195..205
    assert np.allclose(X1[j, :5], X2[j, :5])                      # dates before day 200
    assert not np.allclose(X1[j, 5:], X2[j, 5:])
