import numpy as np
from simulators.rbergomi import rbergomi_paths


def test_hybrid_rbergomi_variance():

    H = 0.07

    S, v, t, Wt = rbergomi_paths(
        N=20_000,
        n_steps=50,
        T=1.0,
        H=H,
        seed=42
    )

    # Initial values
    assert np.all(Wt[:, 0] == 0)

    # Empirical variance across simulated paths
    empirical = Wt.var(axis=0, ddof=1)

    # Theoretical variance
    theoretical = t ** (2 * H)

    # Check selected time points
    indices = [10, 25, 50]

    assert np.allclose(
        empirical[indices],
        theoretical[indices],
        rtol=0.04,
        atol=0.015
    )