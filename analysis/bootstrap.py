"""Block bootstrap of paired reward differences, resampled by date (P5).

Owner: P5 (backtest experiments and statistics). First deliverable, end of day 2: the
statistics toolkit run on a toy example.

To do: confidence intervals, multiple-testing correction.
Per-path rewards of every method on the same test paths are in
results/paths/<experiment>_<problem>_seed<seed>.npz (arrays reward_<method>, tau_<method>);
experiments/paired.py gives the simple iid paired errors to compare against.
"""


def block_bootstrap_diff(diff, block_ids, n_boot=10_000, seed=0):
    """diff: per-window reward difference; block_ids: the date block of each window.
    Return (mean, lower, upper) of a 95% interval."""
    raise NotImplementedError("P5")
