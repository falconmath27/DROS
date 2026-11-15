# Optimal Stopping with Memory

**MAC F244 Stochastic Calculus, BITS Pilani, K K Birla Goa Campus — course project (Suggested Topic 16:
optimal stopping problems and their applications to American option pricing).**

Pricing an American option is an optimal stopping problem, classically solved by backward
induction on a Markovian state. We replicate the optimal stopping policy gradient (OSPG) of
Damera Venkata and Bhattacharyya (NeurIPS 2023), which learns a stopping rule directly with a
memoryless (MLP) or recurrent (GRU) neural policy; extend it to American puts under rough
Bergomi volatility; and test learned exit timing on ten years of real market data from
NIFTY 50, S&P 100 and crypto.

## Deliverables

| | File |
| --- | --- |
| Report (5 pages) | [`report/main.pdf`](report/main.pdf) (source: [`report/main.tex`](report/main.tex)) |
| Slides (15 slides, 15-20 min viva) | [`report/slides.pdf`](report/slides.pdf) (source: [`report/slides.tex`](report/slides.tex)) |
| Source code | this repository (Python); see [Repository structure](#repository-structure) |

## Team

| Name | ID number | Contribution |
| --- | --- | --- |
| Anamay Mittal | 2024ADPS0865G | OSPG implementation and replication; optimal stopping theory, binomial tree, finite-difference PDE solver and baseline methods |
| Vraj Shah | 2024ADPS0846G | Market data pipeline and backtest harness (data loading, cleaning, windows, costs, walk-forward folds, rules) |
| Anshul D Jain | 2024ADPS0844G | Real-data backtest experiments and statistics (full grid, block bootstrap, cost and regime analysis) |
| Sachit Kapoor | 2024ADPS0858G | Final report and presentation |
| Mridul Kwatra | 2024ADPS0819G | Fractional Brownian motion and rough Bergomi simulation; fBm and rough-volatility experiments |

## Key results

1. **Replication.** Against exact prices from a binomial tree and our finite-difference solver
   of the free-boundary problem (which agree to within 4×10⁻⁴), OSPG-MLP is within 0.12% on an
   American put and 0.44% on a 20-asset geometric call, ahead of Longstaff–Schwartz and fitted
   Q-iteration (0.9–1.6%).
2. **Memory in simulation.** When stopping fractional Brownian motion with Hurst parameter
   h ≤ 0.3, the recurrent policy clearly beats the memoryless one; in a reduced rough Bergomi
   pilot no method is distinguishable.
3. **Real data.** After transaction costs, and with date-block bootstrap inference, no learned or
   rule-based exit beats holding each position to its 30-day deadline. On equities OSPG-GRU
   falls short of holding by 0.55–1.40 percentage points per window. Prices drifted upwards over
   the sample, which is consistent with optional sampling for a submartingale.

Full tables: [`results/summary.md`](results/summary.md) (option benchmarks),
[`results/paired.md`](results/paired.md) (paired comparisons),
[`results/p5/summary/`](results/p5/summary/) (backtest statistics) and
[`results/report/`](results/report/) (the tables used in the report and slides).

## Repository structure

```
ospg/          OSPG: policy.py (MLP and GRU policies), loss.py (the paper's Algorithm 1), train.py
baselines/     binomial.py (tree), pde.py (finite-difference free-boundary solver),
               lsm.py (Longstaff–Schwartz), dos.py (Deep Optimal Stopping), fqi.py (fitted Q-iteration)
simulators/    gbm.py (GBM option problems), fbm.py (exact fBm), rbergomi.py (rough Bergomi,
               hybrid scheme), market.py (synthetic GARCH prices)
data/          market data: loaders/, clean.py, windows.py, features.py, splits.py
backtest/      rules.py (hold, take-profit/stop-loss, trailing stop), costs.py, run.py, summarize.py
analysis/      bootstrap.py (paired moving-block bootstrap by date, Holm correction)
common/        problem.py: the shared data format used by every method
experiments/   run_all.py (option benchmarks E1-E5), rough_vol/ (fBm, rough Bergomi),
               backtest/ (real-data grid and statistics), summarize.py, paired.py,
               benchmarks.py (tree vs PDE), report_tables.py (tables for report and slides)
results/       generated outputs (never edited by hand)
report/        main.tex / main.pdf (report), slides.tex / slides.pdf (presentation)
tests/         80 tests: pricing accuracy, loss gradient, causality (no look-ahead), simulators,
               data accounting, bootstrap
docs/          the OSPG paper and the project plan
```

Every method shares one interface: a problem is a pair `X` (features, shape `(N, H+1, d)`) and
`R` (reward for stopping at each date, discounted to date 0, shape `(N, H+1)`); each method has
`fit(X_train, R_train, X_val, R_val)` and `stop(X, R) -> tau`, and at date `j` may use only
`X[:, :j+1]` and `R[:, :j+1]` (checked by the tests).

## How to run

Python 3.12 or later. Run every command from the repository root.

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 80 tests, about a minute
```

Regenerate the tables from the saved results (no training, a few minutes in total):

```bash
python -m experiments.benchmarks               # tree vs PDE ground truth (about 1 min)
python -m experiments.summarize                # option benchmarks -> results/summary.md
python -m experiments.paired                   # paired comparisons -> results/paired.md
python -m backtest.summarize                   # data coverage -> results/p4/
python -m experiments.backtest.summarize       # backtest statistics -> results/p5/summary/ (about 10 min)
python -m experiments.report_tables            # tables for the report and slides -> results/report/
```

Re-run the experiments (long; fixed seeds throughout):

```bash
python -m experiments.run_all E1 E2 E3         # option benchmarks (3-4 h; E3 needs about 2.5 GB RAM)
python -m experiments.rough_vol.run_fbm --medium
python -m experiments.rough_vol.run_e4
python -m experiments.backtest.full_grid       # real-data grid: 3 markets x 2 rewards x 3 cost levels
```

Data download and execution assumptions for the backtest are documented in
[`backtest/P4_GUIDE.md`](backtest/P4_GUIDE.md) and
[`experiments/backtest/README.md`](experiments/backtest/README.md).

Compile the report and slides from inside `report/` with `pdflatex main.tex` (twice) and
`pdflatex slides.tex`, or with `tectonic main.tex` and `tectonic slides.tex`.

**Note on line endings.** The backtest statistics check SHA-256 hashes of their frozen inputs.
On Windows, clone with `git config core.autocrlf false`; otherwise git rewrites line endings and
`experiments.backtest.summarize` stops with a hash mismatch.

## Implementation notes

- All option values are out of sample (trained on one set of paths, evaluated on another), so
  they are lower bounds on the true price.
- OSPG needed three practical choices that the paper does not describe: an initial "mostly
  wait" output bias, a small penalty on the pre-sigmoid outputs (without it the GRU policy
  stalled at "never stop"), and a separate decision at date 0, where all simulated paths share
  one state.
- OSPG-MLP uses batch normalisation as in the paper; the baselines run without it because it
  lowered fitted Q-iteration's accuracy on the American put.
- The OSPG paper's own max-call values lie below Becker et al.'s lower bound, so the
  replication compares against Becker et al. (2019).
- The rough-volatility results are reduced-compute runs (fewer paths and splits, shorter
  training); `results/summary.md` also contains an earlier full-budget rough Bergomi run made
  with an exact Cholesky simulator, which the current hybrid-scheme code no longer reproduces.
- Market universes use current index membership (survivorship bias), execution is at daily
  closes with assumed costs, and overlapping trade windows are not a portfolio return.

## References

- N. Damera Venkata and C. Bhattacharyya. Deep recurrent optimal stopping. *NeurIPS*, 2023.
- S. Becker, P. Cheridito and A. Jentzen. Deep optimal stopping. *JMLR* 20(74), 2019.
- F. A. Longstaff and E. S. Schwartz. Valuing American options by simulation: a simple
  least-squares approach. *Review of Financial Studies* 14(1), 2001.
- C. Bayer, P. Friz and J. Gatheral. Pricing under rough volatility. *Quantitative Finance* 16(6), 2016.
- J. Gatheral, T. Jaisson and M. Rosenbaum. Volatility is rough. *Quantitative Finance* 18(6), 2018.
- M. Bennedsen, A. Lunde and M. S. Pakkanen. Hybrid scheme for Brownian semistationary
  processes. *Finance and Stochastics* 21(4), 2017.
- P. A. Forsyth and K. R. Vetzal. Quadratic convergence for valuing American options using a
  penalty method. *SIAM Journal on Scientific Computing* 23(6), 2002.
- Data: Yahoo Finance (via `yfinance`) for NIFTY 50 and S&P 100 constituents; Binance public spot
  prices and CoinGecko market-cap rankings for crypto.
