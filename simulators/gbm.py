"""Geometric Brownian motion (exact simulation) and the GBM option problems
used in the replication.

Under the risk-neutral measure each asset follows
    dS^m_t = (r - delta) S^m_t dt + sigma S^m_t dW^m_t,
with correlation rho between the Brownian motions, so exactly
    S_{t+dt} = S_t exp((r - delta - sigma^2/2) dt + sigma sqrt(dt) Z).
"""
import numpy as np

from common.problem import Problem


def gbm_log_chunks(r, sigma, T, H, N, d=1, delta=0.0, rho=0.0, seed=0):
    """Yield (start, log(S/S0)) for consecutive blocks of paths, shape (n, H+1, d), so that
    large problems never hold all paths in float64 at once."""
    rng = np.random.default_rng(seed)
    dt = T / H
    corr = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    L = np.linalg.cholesky(corr)
    chunk = max(1, 2_000_000 // (H * d))
    for a in range(0, N, chunk):
        n = min(chunk, N - a)
        Z = rng.standard_normal((n, H, d)) @ L.T      # correlated normals
        steps = (r - delta - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        out = np.zeros((n, H + 1, d))
        np.cumsum(steps, axis=1, out=out[:, 1:])
        yield a, out


def gbm_paths(S0, r, sigma, T, H, N, d=1, delta=0.0, rho=0.0, seed=0):
    """Return prices of shape (N, H+1, d) at the dates t_j = j T / H."""
    out = np.empty((N, H + 1, d))
    for a, logs in gbm_log_chunks(r, sigma, T, H, N, d, delta, rho, seed):
        out[a:a + len(logs)] = logs
    np.exp(out, out=out)
    out *= np.asarray(S0, dtype=float)
    return out


def american_put_problem(S0=36.0, K=40.0, r=0.06, sigma=0.2, T=1.0, H=50, N=100_000, seed=0):
    """1-D put (Longstaff-Schwartz test case). Bermudan with H exercise dates after 0."""
    S = gbm_paths(S0, r, sigma, T, H, N, seed=seed)
    t = np.arange(H + 1) * T / H
    R = np.exp(-r * t) * np.maximum(K - S[:, :, 0], 0.0)
    X = S / K
    return Problem(X, R, name=f"american_put_S0={S0}", itm_only=True,
                   info=dict(S0=S0, K=K, r=r, sigma=sigma, T=T, H=H))


def bermudan_maxcall_problem(d=20, s0=100.0, N=40_000, seed=0,
                             K=100.0, r=0.05, delta=0.1, sigma=0.2, T=3.0, H=9):
    """Paper Section 5 / Becker et al.: payoff (max_m S^m - K)^+, independent assets."""
    S = gbm_paths(s0, r, sigma, T, H, N, d=d, delta=delta, rho=0.0, seed=seed)
    t = np.arange(H + 1) * T / H
    smax = S.max(axis=2)
    R = np.exp(-r * t) * np.maximum(smax - K, 0.0)
    # Features: every asset, sorted descending (the payoff is symmetric in assets),
    # plus the maximum itself.
    Ssorted = -np.sort(-S, axis=2)
    X = np.concatenate([Ssorted, smax[:, :, None]], axis=2) / K
    return Problem(X, R, name=f"maxcall_d={d}_s0={s0}", itm_only=True,
                   info=dict(d=d, s0=s0, K=K, r=r, delta=delta, sigma=sigma, T=T, H=H))


def geometric_call_problem(d=7, s0=100.0, N=100_000, seed=0,
                           K=100.0, r=0.0, delta=0.02, sigma=0.25, rho=0.75, T=2.0, H=100):
    """Paper Appendix D.3: call on the geometric average of d correlated assets.
    Built block by block, with features stored in float32: with d = 20, N = 100,000 and
    100 dates the float64 version needed several GB and ran out of memory."""
    t = np.arange(H + 1) * T / H
    X = np.empty((N, H + 1, d + 1), dtype=np.float32)
    G = np.empty((N, H + 1))
    for a, logs in gbm_log_chunks(r, sigma, T, H, N, d, delta, rho, seed):
        n = len(logs)
        G[a:a + n] = s0 * np.exp(logs.mean(axis=2))
        X[a:a + n, :, :d] = s0 * np.exp(logs) / K
        X[a:a + n, :, d] = G[a:a + n] / K
    R = np.exp(-r * t) * np.maximum(G - K, 0.0)
    return Problem(X, R, name=f"geocall_d={d}", itm_only=True,
                   info=dict(d=d, s0=s0, K=K, r=r, delta=delta, sigma=sigma, rho=rho, T=T, H=H))


def geometric_call_reference(d, s0=100.0, K=100.0, r=0.0, delta=0.02, sigma=0.25, rho=0.75,
                             T=2.0, n=20_000):
    """Semi-analytic American price: the geometric average of correlated GBMs is itself a
    GBM with volatility sigma_G and dividend yield delta_G, so the d-asset American call
    equals a 1-D American call, priced here with a fine binomial tree."""
    from baselines.binomial import binomial_price
    sigma_G = sigma * np.sqrt((1 + (d - 1) * rho) / d)
    delta_G = delta + 0.5 * sigma**2 - 0.5 * sigma_G**2
    return binomial_price(s0, K, r, sigma_G, T, n, kind="call", delta=delta_G)
