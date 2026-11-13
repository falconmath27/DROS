"""STAND-IN rough Bergomi simulator (for testing P2's baselines until P3's version is ready).

Model (Bayer, Friz and Gatheral, 2016), risk-neutral:
    dS_t = r S_t dt + sqrt(v_t) S_t dZ_t,     Z = rho W + sqrt(1 - rho^2) W_perp
    v_t  = xi0 exp(eta Wt_t - eta^2/2 t^{2H})
    Wt_t = sqrt(2H) int_0^t (t - s)^{H - 1/2} dW_s     (Riemann-Liouville fBm)

Simulation is exact on the grid for (Wt, W): the 2n-dimensional Gaussian vector
(Wt_{t_1..t_n}, W_{t_1..t_n}) is drawn from its covariance by Cholesky factorisation.
The price uses a left-point (Euler) step in log-space, which is an exact martingale.
"""
import numpy as np
from scipy.special import hyp2f1

from common.problem import Problem


def _joint_covariance(t, H):
    """Covariance of (Wt_{t_1..t_n}, W_{t_1..t_n}) for grid t (all > 0)."""
    n = len(t)
    a = H + 0.5
    C = np.zeros((2 * n, 2 * n))
    u = np.minimum.outer(t, t)
    v = np.maximum.outer(t, t)
    # Cov(Wt_u, Wt_v) = 2H/(H+1/2) u^{H+1/2} v^{H-1/2} 2F1(1/2-H, 1; 3/2+H; u/v)
    C[:n, :n] = 2 * H / a * u**a * v**(H - 0.5) * hyp2f1(0.5 - H, 1.0, 1.5 + H, u / v)
    # Cov(W_u, W_v) = min(u, v)
    C[n:, n:] = u
    # Cov(Wt_{t_i}, W_{t_k}) = sqrt(2H)/(H+1/2) (t_i^{H+1/2} - (t_i - min(t_i, t_k))^{H+1/2})
    ti = t[:, None]
    mk = np.minimum(t[:, None], t[None, :])
    C[:n, n:] = np.sqrt(2 * H) / a * (ti**a - (ti - mk) ** a)
    C[n:, :n] = C[:n, n:].T
    return C


def rbergomi_paths_exact(N=100_000, n_steps=100, T=1.0, H=0.07, eta=1.9, rho=-0.9, xi0=0.09,
                   r=0.05, S0=100.0, seed=0):
    """Return (S, v) each of shape (N, n_steps+1) on the grid t_k = k T / n_steps."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.arange(1, n_steps + 1) * dt
    C = _joint_covariance(t, H)
    L = np.linalg.cholesky(C + 1e-12 * np.eye(2 * n_steps))
    G = rng.standard_normal((N, 2 * n_steps)) @ L.T
    Wt = G[:, :n_steps]
    W = G[:, n_steps:]
    tt = np.concatenate([[0.0], t])
    Wt = np.hstack([np.zeros((N, 1)), Wt])
    v = xi0 * np.exp(eta * Wt - 0.5 * eta**2 * tt ** (2 * H))          # (N, n+1)
    dW = np.diff(np.hstack([np.zeros((N, 1)), W]), axis=1)            # (N, n)
    dZ = rho * dW + np.sqrt(1 - rho**2) * np.sqrt(dt) * rng.standard_normal((N, n_steps))
    vl = v[:, :-1]                                                    # left point
    logS = np.cumsum((r - 0.5 * vl) * dt + np.sqrt(vl) * dZ, axis=1)
    S = S0 * np.exp(np.hstack([np.zeros((N, 1)), logS]))
    return S, v, tt, Wt

def rbergomi_paths(
    N=100_000, n_steps=100, T=1.0, H=0.07,
    eta=1.9, rho=-0.9, xi0=0.09,
    r=0.05, S0=100.0, seed=0
):
    """Rough Bergomi simulation using the hybrid scheme (kappa=1)."""

    if not 0 < H <= 0.5:
        raise ValueError("Rough Bergomi requires 0 < H <= 0.5")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    tt = np.linspace(0, T, n_steps + 1)

    # Brownian increments
    dW = np.sqrt(dt) * rng.standard_normal((N, n_steps))

    # Volterra process
    Wt = np.zeros((N, n_steps + 1))

    if np.isclose(H, 0.5):
        # At H=0.5, the Volterra process is ordinary Brownian motion
        Wt[:, 1:] = np.cumsum(dW, axis=1)

    else:
        alpha = H - 0.5

        # Generate the near-diagonal Wiener integrals.
        # These are correlated with the corresponding Brownian increments.

        cov = dt**(alpha + 1) / (alpha + 1)
        var = dt**(2 * alpha + 1) / (2 * alpha + 1)

        conditional_mean = (cov / dt) * dW
        conditional_var = max(var - cov**2 / dt, 0.0)

        J = (
            conditional_mean
            + np.sqrt(conditional_var)
            * rng.standard_normal((N, n_steps))
        )

        # Optimal evaluation points for the remaining kernel terms
        k = np.arange(2, n_steps + 1)

        b_star = (
            (k**(alpha + 1) - (k - 1)**(alpha + 1))
            / (alpha + 1)
        )**(1 / alpha)

        weights = (b_star * dt)**alpha

        # Build the convolution matrix
        kernel = np.zeros((n_steps, n_steps))

        for lag, weight in zip(k, weights):
            rows = np.arange(lag, n_steps + 1)
            cols = rows - lag
            kernel[rows - 1, cols] = weight

        # Hybrid approximation of the Volterra process
        Wt[:, 1:] = np.sqrt(2 * H) * (
            J + dW @ kernel.T
        )

    # Rough variance process
    v = xi0 * np.exp(
        eta * Wt - 0.5 * eta**2 * tt**(2 * H)
    )

    # Correlated Brownian motion driving the stock price
    dZ = (
        rho * dW
        + np.sqrt(1 - rho**2)
        * np.sqrt(dt)
        * rng.standard_normal((N, n_steps))
    )

    # Log-Euler stock price simulation
    vl = v[:, :-1]

    logS = np.cumsum(
        (r - 0.5 * vl) * dt + np.sqrt(vl) * dZ,
        axis=1
    )

    S = S0 * np.exp(
        np.hstack([np.zeros((N, 1)), logS])
    )

    return S, v, tt, Wt
def rbergomi_put_problem(N=100_000, n_steps=100, exercise_every=2, K=100.0, lags=0, seed=0,
                         **model):
    """American (Bermudan) put under rough Bergomi.

    Features at an exercise date: S/K and v/xi0. With lags=k, also the values of S/K and
    v/xi0 at the k previous exercise dates (a hand-made 'memory' for Markovian methods).
    """
    params = dict(T=1.0, H=0.07, eta=1.9, rho=-0.9, xi0=0.09, r=0.05, S0=100.0)
    params.update(model)
    S, v, tt, _ = rbergomi_paths(N=N, n_steps=n_steps, seed=seed, **params)
    S, v, tt = S[:, ::exercise_every], v[:, ::exercise_every], tt[::exercise_every]
    R = np.exp(-params["r"] * tt) * np.maximum(K - S, 0.0)
    base = np.stack([S / K, v / params["xi0"]], axis=2)
    feats = [base]
    for k in range(1, lags + 1):
        lagged = np.concatenate([np.repeat(base[:, :1], k, axis=1), base[:, :-k]], axis=1)
        feats.append(lagged)
    X = np.concatenate(feats, axis=2)
    return Problem(X, R, name=f"rbergomi_put_lags={lags}", itm_only=True,
                   info=dict(K=K, exercise_every=exercise_every, n_steps=n_steps, **params))
