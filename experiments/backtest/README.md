# Backtest experiments (P5)

## Regeneration order (PowerShell, repository root)

1. Validate: `python -B -m pytest -q -p no:cacheprovider --basetemp "$env:TEMP\dros-p5-tests-$PID" tests`
2. Integration, only if needed: `python -B -m experiments.backtest.run`
3. Complete/resume training, only if coverage is incomplete: `python -B -m experiments.backtest.full_grid --workers 4 --progress-interval 10`
4. Summarize completed frozen artifacts: `python -B -m experiments.backtest.summarize --n-boot 2000`
5. Compile from report: `cd report`, then `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice and `pdflatex -interaction=nonstopmode -halt-on-error slides.tex` twice.

The summary command does not train models. Its default output is results/p5/summary:
market_cost_tables.csv (both objectives, both metrics, all costs), regime_tables.csv
(2020/2022/other calendar slices), paired_statistics.csv and .json (intervals, Holm
p-values, seed and 15/30/60-day sensitivity), persistence.csv and persistence.png,
primary.tex, markets.tex, interpretation.tex, findings.txt and manifest.json.
The manifest hashes consumed artifacts and generated outputs. Primary estimates
average seeds per window and weight windows equally. Holm families are corrected
within each market/objective/cost/metric, not across the entire study.
The persistence plot uses training-only absolute-return lag-one autocorrelation.
Calendar slices are descriptive; they are not Task 5's unfinished Hurst or
calm/stress classification. No 2008 coverage is claimed. Undefined autocorrelations
remain missing in persistence.csv and do not appear on the plot.

Real E5 uses the P4 harness. Older synthetic artifacts under results/paths/ are
separate and never loaded by summarize.py. Utility scores and economic net returns
are separately labelled; neither represents an annualized portfolio return.
PDF compilation requires a LaTeX distribution with beamer, graphicx and booktabs.
P4 review remains pending until the teammate reviews report/backtest_results.tex.

Portable compiler used on this machine (run from the repository root):

    & 'C:\Users\ASUS\Documents\Codex\work\p5-tex\tectonic.exe' --keep-logs report/main.tex
    & 'C:\Users\ASUS\Documents\Codex\work\p5-tex\tectonic.exe' --keep-logs report/slides.tex

These commands write report/main.pdf and report/slides.pdf. The first invocation
downloads standard TeX packages. costs.tex and regimes.tex are compact report
tables generated from the full CSV results. report/P4_REVIEW_REQUEST.md is a draft
to forward to P4; it is not a completed review.

config.json freezes the three core markets, their P4 input hashes, features, venue
cost assumptions, folds, seeds, methods, objectives and planned contrasts.

## One-fold integration check

    python -B -m experiments.backtest.run

This writes shortened-training validation artifacts to results/p5/one_fold/.

## Full grid and cost sensitivity

    python -B -m experiments.backtest.full_grid

The full runner uses the configured training budgets across every fold and seed for
NIFTY 50, S&P 100 and crypto, both objectives, and 0x/1x/2x venue costs. Each cost
scenario rebuilds rewards and refits all learned policies. It checkpoints every method,
resumes completed work, records failures, and writes coverage.json and coverage.csv
under results/p5/full_grid/. It defaults to six independent fold/seed workers and one
Torch thread per worker. Set --workers to a smaller value if memory is constrained.
While workers are training it prints a heartbeat every 10 seconds, then groups every
three completed fold/seed jobs into a batch summary with method counts, elapsed time
and ETA. These can be adjusted with --progress-interval and --progress-batch-size:

    python -B -m experiments.backtest.full_grid --progress-interval 10 --progress-batch-size 3

For a quick orchestration check that stays separate from final results:

    python -B -m experiments.backtest.full_grid --smoke --markets sp100 --objectives linear --cost-multipliers 1 --folds 0 --seeds 0

Use filters to divide final work across terminals or machines. Give each process a
different --output-root, then combine only after verifying its coverage.json; two
processes must not write the same scenario directory concurrently.

The configured equity constituents and crypto ranking are current/retrieval-date
universes rather than historical point-in-time membership. Results therefore carry
survivorship and selection bias. Entry and exit use adjusted daily-close reference
prices with assumed friction, which is a stylized execution model rather than a claim
of guaranteed same-close fills. FX and gold remain optional.
