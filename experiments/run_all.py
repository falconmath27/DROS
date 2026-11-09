"""All experiments. Run from the project root:
    python -m experiments.run_all            (all benchmarks and full real-market training)
    python -m experiments.run_all E1 E4      (selected experiments)
E1-E4/synthetic results go to results/raw.jsonl; then run python -m experiments.summarize.
Real E5 outputs go to results/p4; see backtest/P4_GUIDE.md. Full E5 training can be long.
"""
import sys

from experiments.runner import run
from simulators.gbm import (american_put_problem, bermudan_maxcall_problem,
                            geometric_call_problem, geometric_call_reference)
from simulators.market import make_exit_windows, synthetic_prices, walk_forward_split
from simulators.rbergomi import rbergomi_put_problem


def E1_american_put(seeds=(0, 1, 2)):
    """P2 benchmark: 1-D put, S0=36, K=40, r=6%, sigma=20%, T=1, 50 dates.
    Exact Bermudan price (binomial tree) = 4.478."""
    for s in seeds:
        splits = american_put_problem(N=100_000, seed=100 + s).random_split(seed=s)
        run("E1", "put_S0=36", splits, ["LSM", "DOS", "FQI-MLP", "OSPG-MLP", "OSPG-GRU"], s,
            extra=dict(reference=4.478))


def E2_bermudan_maxcall(seeds=(0, 1, 2)):
    """Paper Table 1, d in {20, 50}, s0 = 100 (reduced grid for the one-week plan)."""
    for d in (20, 50):
        for s in seeds:
            splits = bermudan_maxcall_problem(d=d, s0=100.0, N=40_000,
                                              seed=200 + s).random_split(seed=s)
            run("E2", f"maxcall_d={d}", splits,
                ["LSM", "DOS", "FQI-MLP", "FQI-GRU", "OSPG-MLP", "OSPG-GRU"], s)


def E3_geometric_call(seeds=(0, 1, 2)):
    """Paper Table 4: American geometric-average call, 100 exercise dates.
    OSPG-GRU is included to check that the recurrent policy loses nothing in this
    Markovian problem."""
    for d in (7, 20):
        ref = geometric_call_reference(d)
        for s in seeds:
            splits = geometric_call_problem(d=d, N=100_000, seed=300 + s).random_split(seed=s)
            run("E3", f"geocall_d={d}", splits, ["LSM", "FQI-MLP", "OSPG-MLP", "OSPG-GRU"], s,
                extra=dict(reference=ref))
            del splits


def E4_rough_bergomi(seeds=(0, 1)):
    """STAND-IN rough Bergomi American put (replace with P3's simulator when ready).
    LSM is also run with 3 lagged features ('hand-made memory')."""
    from baselines.lsm import LSM
    for s in seeds:
        base = rbergomi_put_problem(N=100_000, lags=0, seed=400 + s)
        splits = base.random_split(seed=s)
        run("E4", "rbergomi_put", splits,
            ["LSM", "DOS", "FQI-GRU", "OSPG-MLP", "OSPG-GRU"], s)
        lagged = rbergomi_put_problem(N=100_000, lags=3, seed=400 + s).random_split(seed=s)
        run("E4", "rbergomi_put", lagged, ["LSM-lags3"], s,
            method_overrides={"LSM-lags3": lambda: LSM()})


def E5_synthetic_market(seeds=(0, 1)):
    """STAND-IN market data (synthetic GARCH prices with weak momentum), walk-forward.
    Replace synthetic_prices(...) with load_prices_csv(path) for real data."""
    for s in seeds:
        prices = synthetic_prices(n_assets=20, n_days=3000, seed=500 + s)
        X, R, start = make_exit_windows(prices, H=20, cost=0.001, stride=2)
        splits = walk_forward_split(X, R, start, H=20)
        run("E5", "synthetic_market", splits,
            ["Hold", "LSM", "DOS", "OSPG-MLP", "OSPG-GRU"], s)


def E5_market(seeds=(0, 1), markets=("nifty50", "sp100", "crypto")):
    """Real-market P4 harness; outputs isolated under results/p4, with dated metadata.

    Full learned-model training belongs to P5's experiments and can take substantial
    time. For a quick handoff check use python -m backtest.run --smoke --max-folds 1.
    """
    from backtest.run import main
    for market in markets:
        args = ["--market", market, "--seeds", *map(str, seeds), "--methods",
                "Hold", "TakeProfitStopLoss", "TrailingStop", "LSM", "DOS", "OSPG-MLP", "OSPG-GRU"]
        if market == "crypto":
            args += ["--symbols", "top10", "--start", "2017-01-01"]
        main(args)


ALL = dict(E1=E1_american_put, E2=E2_bermudan_maxcall, E3=E3_geometric_call,
           E4=E4_rough_bergomi, E5=E5_market)

if __name__ == "__main__":
    for key in (sys.argv[1:] or list(ALL)):
        print(f"===== {key}", flush=True)
        (E5_synthetic_market if key == "E5synthetic" else ALL[key])()
