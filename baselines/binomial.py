"""Cox-Ross-Rubinstein binomial tree: exact-in-the-limit prices for 1-D options.

Backward induction (Snell envelope) on a recombining tree:
    value at a node = max(exercise value, discounted risk-neutral average of the children).
"""
import math

import numpy as np


def binomial_price(S0, K, r, sigma, T, n, kind="put", delta=0.0,
                   american=True, exercise_every=1):
    """Price a put or call on one asset.

    n               number of tree steps
    delta           continuous dividend yield
    american        False -> European (no early exercise)
    exercise_every  exercise allowed only every this many steps (Bermudan);
                    1 -> every step (approximately American)
    """
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - delta) * dt) - d) / (u - d)      # risk-neutral up probability
    assert 0.0 < p < 1.0, "time step too large for these parameters"
    disc = math.exp(-r * dt)

    def payoff(S):
        return np.maximum(K - S, 0.0) if kind == "put" else np.maximum(S - K, 0.0)

    k = np.arange(n + 1)
    V = payoff(S0 * u**k * d**(n - k))                  # values at expiry
    for m in range(n - 1, -1, -1):
        k = np.arange(m + 1)
        V = disc * (p * V[1:] + (1 - p) * V[:-1])       # value of waiting
        if american and m % exercise_every == 0:
            V = np.maximum(V, payoff(S0 * u**k * d**(m - k)))
    return float(V[0])


def exercise_boundary(S0, K, r, sigma, T, n, delta=0.0):
    """For an American put: at each step, the highest node price at which exercising is
    optimal (nan where no node exercises). Returns (times, boundary)."""
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt)); d = 1.0 / u
    p = (math.exp((r - delta) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    k = np.arange(n + 1)
    V = np.maximum(K - S0 * u**k * d**(n - k), 0.0)
    bnd = np.full(n, np.nan)
    for m in range(n - 1, -1, -1):
        k = np.arange(m + 1)
        S = S0 * u**k * d**(m - k)
        cont = disc * (p * V[1:] + (1 - p) * V[:-1])
        ex = K - S
        mask = (ex >= cont) & (ex > 0)
        if mask.any():
            bnd[m] = S[mask].max()
        V = np.maximum(cont, ex)
    return np.arange(n) * dt, bnd


def black_scholes_put(S0, K, r, sigma, T, delta=0.0):
    """European put, closed form."""
    from math import erf, exp, log, sqrt
    N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    d1 = (log(S0 / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * N(-d2) - S0 * exp(-delta * T) * N(-d1)


def perpetual_put(S, K, r, sigma):
    """Perpetual American put (Shreve II, Sec. 8.3): returns (value, boundary L*)."""
    L = 2 * r * K / (2 * r + sigma**2)
    if S <= L:
        return K - S, L
    return (K - L) * (S / L) ** (-2 * r / sigma**2), L
