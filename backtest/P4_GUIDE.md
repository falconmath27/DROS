# P4: real-data harness and handoff

Run from the project root using Python 3.12+. The local `.venv` includes the dependencies.
No brokerage connection, account credentials, orders, or paid data are used.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m pytest -q tests
```

## Reproduce the P4 data and rules

```powershell
.venv/Scripts/python.exe -m backtest.run --market nifty50 --methods Hold TakeProfitStopLoss TrailingStop --output results/p4/nifty50
.venv/Scripts/python.exe -m backtest.run --market sp100 --methods Hold TakeProfitStopLoss TrailingStop --output results/p4/sp100
.venv/Scripts/python.exe -m backtest.run --market crypto --symbols top10 --start 2017-01-01 --methods Hold TakeProfitStopLoss TrailingStop --output results/p4/crypto
```

Default end is **2025-01-01 exclusive**, deliberately frozen; set `--end` explicitly to
extend it. Default equities start is 2015-01-01. Crypto starts later in this example.
Raw requests and current-universe snapshots are cached under `data/cache/` (gitignored).
`--offline` replays exactly matching price requests; universe snapshots must also exist.
`--refresh` fetches again. Partial failures are visible in source manifests; no synthetic
series are silently substituted. To replay independently of the provider, retain the
clean price CSV and run `--market csv --csv <path> --venue us --context-csv <path>`.

The explicit `--output` commands replace the artifacts in that chosen run directory.
Omit `--output` for directories keyed by configuration and actual price/context hashes.
Original `results/raw.jsonl`, replication results, and synthetic-market results are preserved.

## Handoff files in each run directory

| File | Purpose |
| --- | --- |
| `clean_prices.csv` | Audited, adjusted price panel; no currency aggregation |
| `cleaning_log.json` | Invalid prices, missing sessions, retained jumps, coverage and exclusions |
| `source_manifest.json` | Provider, query dates, retrieval date, requested symbols, failures |
| `context.csv`, `context_manifest.json` | Optional local index/VIX source data and failures |
| `dataset_manifest.json`, `run_config.json` | Assumptions, features, universe, actual data hashes and parameters |
| `windows.npz` | `X`, objective `R`, and metadata; load with `MarketWindows.load()` |
| `windows.csv` | Asset, entry, deadline, terminal-index and window-ID mapping |
| `folds.json` | Exact chronological fold boundaries and sample sizes |
| `fold_<n>_seed_<s>.npz` | Same test windows for all methods; rewards, net returns, tau and actual exit dates |
| `metrics.csv`, `summary.md` | Out-of-sample per-window averages; no iid significance claims |

```python
from data.windows import MarketWindows
from data.splits import walk_forward_folds
ds = MarketWindows.load("results/p4/sp100/windows.npz")
folds = walk_forward_folds(ds)
train, validation, test = (x.problem for x in
    (folds[0].train, folds[0].validation, folds[0].test))
# Existing methods: fit(train.X, train.R, validation.X, validation.R)
```

The feature order is saved. First three are return since entry, running drawdown,
remaining-time fraction; then daily log return, trailing volatility, 1/2/5-observation
lagged returns and optional context. Values use data up to the current session only.
External levels are shifted one source observation BEFORE backward/as-of alignment.
NIFTY context uses `^NSEI`/`^INDIAVIX`; US uses `^GSPC`/`^VIX`. Partial source failures
are disclosed. `--context none` disables them; `--context-csv` accepts custom sources.
Do not join global markets by unlagged nominal date. Equity volatility annualization
is 252 sessions, crypto 365 days; this is a descriptive feature, not a return forecast.

## Economic and temporal contract

- One long-only unit position per asset/window, mandatory exit by its deadline; default
  horizon is 30 calendar days, on the last observed panel session on/before the deadline.
  `--sessions 20` explicitly selects a fixed-session alternative.
- Equity and crypto calendars stay separate. Variable numbers of sessions are padded by
  repeating the terminal feature, reward and date to 31 states. The runner clips `tau`
  to `terminal_index`; no additional reward or time after the deadline is possible.
- Entry and exit use the adjusted daily-close reference price, observed at each decision,
  plus proportional friction. This is a stylized close-mark experiment, not a claim that
  an order after observing the close fills at that close. Actual next-open execution would
  require separating unobserved execution rewards from the existing policies' reward input.
- Net return = exit/entry * (1 - sell friction)/(1 + buy friction) - 1 - funding.
  Fees, half-spread and slippage apply on BOTH legs. Funding is a simple initial-notional
  daily scenario, not a leveraged margin-account simulation. Spot funding defaults to zero.
  Optional discounting discounts net reward by actual elapsed calendar days.
- NSE/US/crypto presets are explicitly assumed all-in fee scenarios, not official tariffs:
  fee bps per side 12/1/10, half-spread 2/1/2, slippage 2/1/2 respectively.
  `--cost-json` accepts all `CostModel` fields; `--cost-multiplier 0` or `2` gives sensitivity.
- Hold always reaches the deadline. Profit/loss rule exits at +5%/-3%; trailing stop at
  5% drawdown. Thresholds are fixed ex ante. These are daily-close observations, not
  intraday stop orders inferred from OHLC bars. No test-set threshold tuning.
- Split all assets on common dates. Default expanding folds: 36 months training, 6 months
  validation, 6 months test, advancing 6 months. Windows crossing either boundary are
  purged using the entire scheduled outcome horizon. `--embargo-days` adds an extra gap.
  Source history may supply causal trailing test features; preprocessing fits training only.
- Current/retrieval-date equity membership and crypto rankings have survivorship/selection
  bias. They are not historical constituent reconstructions. Short-history/failed assets
  are logged. Missing delisting proceeds cannot be recovered from close data.
- Missing prices stay missing by default; windows crossing them drop. No interpolation,
  backward filling, or extrapolation. The cleaner flags large moves without deleting crashes.
- Averaging overlapping trade windows is not a self-financing portfolio or CAGR. P5 must
  use date-block bootstrap and retain cross-asset dependence; the harness does not report
  misleading independent-window significance.
- Existing learned models use a global date-zero choice developed for simulated common
  initial states. This limitation is retained and disclosed for heterogeneous market entries.

## Learned-model and P5 runs

```powershell
# Handoff smoke: short neural training, explicitly not research-quality estimates
.venv/Scripts/python.exe -m backtest.run --market sp100 --symbols AAPL MSFT --start 2020-01-01 --end 2025-01-01 --train-months 24 --max-folds 1 --methods Hold LSM DOS FQI-MLP FQI-GRU OSPG-MLP OSPG-GRU --smoke
# Full learning uses the original model training budgets when --smoke is omitted
.venv/Scripts/python.exe -m backtest.run --market sp100 --methods Hold TakeProfitStopLoss TrailingStop LSM OSPG-MLP OSPG-GRU --seeds 0 1 2
# Risk-sensitive objective (P5 chooses/tunes aversion using validation only)
.venv/Scripts/python.exe -m backtest.run --market crypto --reward exponential --risk-aversion 5 --cost-multiplier 2 --prepare-only
```

Exponential reward is `(1-exp(-a*r))/a`; raw economic net returns are saved separately
from utility in `net_return_<method>`. P5 owns final training grids, uncertainty,
multiple comparisons, Hurst/regime analysis, and conclusions. P4 supplies both rewards,
cost sweeps, dated folds and all asset-level observations required for those analyses.

`python -m experiments.run_all E5` now calls the full real-data harness across the three
markets. The original synthetic check is available as `E5synthetic`. Real-data results
live under `results/p4/` and are not mixed into the old replication summarizer.

Optional FX/gold can be supplied through adjusted-close CSVs or explicit Yahoo symbols;
specify an appropriate venue cost JSON and disclose quote-currency/roll assumptions.
They are outside the reduced core universe, as allowed by the project plan.
