"""Finite-difference pricer for the Black-Scholes free-boundary problem (P2).

The value V(t, S) of an American option solves the linear complementarity problem

    min( -dV/dt - L V,  V - g ) = 0,     L V = (r - delta) S V_S + 1/2 sigma^2 S^2 V_SS - r V,

with V(T, S) = g(S): where waiting is better, V satisfies the Black-Scholes PDE (the
Feynman-Kac equation of the discounted value); where exercising is better, V = g.
The boundary between the two regions is the free (early-exercise) boundary.

Numerics: log-price grid x = log S (constant coefficients), Crank-Nicolson in time with a
few fully implicit start-up steps (Rannacher smoothing of the payoff kink), and the
early-exercise constraint V >= g imposed by a penalty iteration (Forsyth and Vetzal,
2002): at each step the tridiagonal system is solved with a large penalty on the nodes
where V < g, until the set of those nodes stops changing.
A Bermudan option (exercise only every few time steps) is priced by solving the
European PDE between exercise dates and taking max(V, g) on each exercise date.
"""
import math

import numpy as np
from scipy.linalg import solve_banded


def _payoff(S, K, kind):
    return np.maximum(K - S, 0.0) if kind == "put" else np.maximum(S - K, 0.0)


def fd_solve(S0, K, r, sigma, T, kind="put", delta=0.0, american=True, exercise_every=None,
             n_x=1000, n_t=1000, width=6.0, rannacher=4, penalty=1e8):
    """Solve the pricing PDE on a grid. Returns a dict with
        price      value at S0 (S0 is a grid node)
        S, V       the grid and the value at time 0
        t, boundary    for american puts/calls: at each time step, the exercise boundary
                       (highest S with V = g for a put, lowest for a call; nan if none)

    american=False         European option (no early exercise)
    exercise_every=m       Bermudan: exercise allowed only every m time steps (and at
                           maturity); None with american=True means every step (American)
    """
    if exercise_every is not None:
        assert n_t % exercise_every == 0, "n_t must be a multiple of exercise_every"
    assert n_x % 2 == 0
    sd = sigma * math.sqrt(T)
    w = max(width * sd, abs(math.log(K / S0)) + 4 * sd)    # half-width in log price
    dx = 2 * w / n_x
    x = math.log(S0) + dx * (np.arange(n_x + 1) - n_x // 2)  # S0 is node n_x // 2
    S = np.exp(x)
    g = _payoff(S, K, kind)
    dt = T / n_t

    a = 0.5 * sigma**2
    b = r - delta - a
    lo = a / dx**2 - b / (2 * dx)            # coefficient of V[i-1]
    di = -2 * a / dx**2 - r                  # coefficient of V[i]
    up = a / dx**2 + b / (2 * dx)            # coefficient of V[i+1]

    def lhs(theta):
        ab = np.zeros((3, n_x + 1))
        ab[0, 2:] = -theta * dt * up         # super-diagonal (rows 1..n_x-1)
        ab[1, :] = 1.0
        ab[1, 1:-1] = 1 - theta * dt * di
        ab[2, :-2] = -theta * dt * lo        # sub-diagonal (rows 1..n_x-1)
        return ab

    ab_cn, ab_im = lhs(0.5), lhs(1.0)
    continuous = american and exercise_every is None

    def edges(tau):
        """Dirichlet values at the two ends of the grid, time to maturity tau."""
        S_lo, S_hi = S[0], S[-1]
        if kind == "put":
            v_lo = K * math.exp(-r * tau) - S_lo * math.exp(-delta * tau)
            if american:
                v_lo = max(v_lo, K - S_lo)
            return v_lo, 0.0
        v_hi = S_hi * math.exp(-delta * tau) - K * math.exp(-r * tau)
        if american:
            v_hi = max(v_hi, S_hi - K)
        return 0.0, v_hi

    V = g.copy()
    bnd = np.full(n_t + 1, np.nan)
    for n in range(n_t):                      # step from tau = n dt to (n+1) dt
        theta, ab = (1.0, ab_im) if n < rannacher else (0.5, ab_cn)
        LV = np.zeros_like(V)
        LV[1:-1] = lo * V[:-2] + di * V[1:-1] + up * V[2:]
        rhs = V + (1 - theta) * dt * LV
        rhs[0], rhs[-1] = edges((n + 1) * dt)
        if continuous:
            Vk = solve_banded((1, 1), ab, rhs)
            active = None
            for _ in range(100):
                pen = np.where(Vk < g, penalty, 0.0)
                pen[0] = pen[-1] = 0.0
                if active is not None and np.array_equal(pen > 0, active):
                    break
                active = pen > 0
                ab2 = ab.copy()
                ab2[1] += pen
                Vk = solve_banded((1, 1), ab2, rhs + pen * g)
            V = Vk
        else:
            V = solve_banded((1, 1), ab, rhs)
            if american and (n + 1) % exercise_every == 0:
                V = np.maximum(V, g)
        if american:
            ex = (V <= g + 1e-6 * K) & (g > 0)
            ex[0] = ex[-1] = False
            if ex.any():
                bnd[n + 1] = S[ex].max() if kind == "put" else S[ex].min()
    if american:                              # limit of the boundary at maturity
        if kind == "put":
            bnd[0] = K * min(1.0, r / delta) if delta > 0 else K
        elif delta > 0:
            bnd[0] = K * max(1.0, r / delta)
    t = T - dt * np.arange(n_t + 1)           # calendar time of each step
    return dict(price=float(V[n_x // 2]), S=S, V=V, t=t[::-1], boundary=bnd[::-1])


def fd_price(S0, K, r, sigma, T, kind="put", delta=0.0, american=True, exercise_every=None,
             n_x=1000, n_t=1000):
    """Price only (see fd_solve)."""
    return fd_solve(S0, K, r, sigma, T, kind, delta, american, exercise_every, n_x, n_t)["price"]
