"""Paired date-block bootstrap for the P5 real-data backtests.

Results are aligned before subtraction by market, fold, seed and window ID. Seed-level
differences are averaged for each window, then entry dates are resampled in contiguous
moving blocks. Every asset observed on a sampled date is retained. Blocks are built and
sampled independently inside each market/fold, so they never cross walk-forward test
boundaries. Replicate means are pooled with one observation per resampled window.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


PAIR_KEYS = ["market", "fold", "seed", "window_id"]
WINDOW_KEYS = ["market", "fold", "window_id"]
REQUIRED_RESULT_COLUMNS = PAIR_KEYS + ["entry_date", "asset_id", "method", "score"]


def load_fold_results(path, *, market=None, fold=None, seed=None, methods=None,
                      metric="objective"):
    """Load one P4/P5 fold artifact into a validated long-form result table.

    metric may be objective, reward or net_return. P4 artifacts store objective and
    reward identically; objective loading falls back to reward for older artifacts.
    """
    path = Path(path)
    match = re.search(r"fold_(\d+)_seed_(\d+)\.npz$", path.name)
    if fold is None:
        if match is None:
            raise ValueError("fold must be supplied when it cannot be read from the filename")
        fold = int(match.group(1))
    if seed is None:
        if match is None:
            raise ValueError("seed must be supplied when it cannot be read from the filename")
        seed = int(match.group(2))
    if metric not in {"objective", "reward", "net_return"}:
        raise ValueError("metric must be objective, reward or net_return")

    with np.load(path, allow_pickle=False) as data:
        required = {"window_id", "entry_date", "asset_id", "market"}
        missing_metadata = required - set(data.files)
        if missing_metadata:
            raise ValueError(f"{path} is missing metadata: {sorted(missing_metadata)}")
        n = len(data["window_id"])
        for name in required:
            if data[name].shape != (n,):
                raise ValueError(f"{path}: {name} must have shape ({n},)")
        window_id = data["window_id"].astype(str)
        if len(np.unique(window_id)) != n:
            raise ValueError(f"{path} contains duplicate window IDs")

        file_markets = data["market"].astype(str)
        if market is not None and not np.all(file_markets == str(market)):
            raise ValueError(f"{path} contains a market different from {market}")
        market_values = file_markets if market is None else np.full(n, str(market))

        prefixes = [f"{metric}_"]
        if metric == "objective":
            prefixes.append("reward_")
        available = set()
        for prefix in prefixes:
            available.update(name[len(prefix):] for name in data.files
                             if name.startswith(prefix))
        selected = sorted(available) if methods is None else list(methods)
        unavailable = set(selected) - available
        if unavailable:
            raise ValueError(f"{path} is missing {metric} results for {sorted(unavailable)}")

        rows = []
        for method in selected:
            candidates = [f"{prefix}{method}" for prefix in prefixes]
            key = next((name for name in candidates if name in data.files), None)
            score = np.asarray(data[key], dtype=float)
            if score.shape != (n,):
                raise ValueError(f"{path}: {key} must have shape ({n},)")
            if not np.isfinite(score).all():
                raise ValueError(f"{path}: {key} contains non-finite scores")
            rows.append(pd.DataFrame({
                "market": market_values,
                "fold": int(fold),
                "seed": int(seed),
                "window_id": window_id,
                "entry_date": data["entry_date"].astype("datetime64[D]"),
                "asset_id": data["asset_id"].astype(str),
                "method": method,
                "score": score,
            }))
    return pd.concat(rows, ignore_index=True)


def align_method_scores(results, left, right):
    """Align two methods on the full paired key and return left minus right."""
    _require_columns(results, REQUIRED_RESULT_COLUMNS)
    selected = results.loc[results["method"].isin([left, right])].copy()
    if not np.isfinite(selected["score"].to_numpy(dtype=float)).all():
        raise ValueError("Selected method scores contain non-finite values")

    duplicates = selected.duplicated(PAIR_KEYS + ["method"], keep=False)
    if duplicates.any():
        examples = selected.loc[duplicates, PAIR_KEYS + ["method"]].head().to_dict("records")
        raise ValueError(f"Duplicate method results for paired keys: {examples}")

    sides = {}
    for method, suffix in ((left, "left"), (right, "right")):
        side = selected.loc[selected["method"] == method,
                            PAIR_KEYS + ["entry_date", "asset_id", "score"]].copy()
        if side.empty:
            raise ValueError(f"No results found for method {method}")
        sides[suffix] = side.rename(columns={
            "entry_date": f"entry_date_{suffix}",
            "asset_id": f"asset_id_{suffix}",
            "score": f"score_{suffix}",
        })

    paired = sides["left"].merge(sides["right"], on=PAIR_KEYS, how="outer",
                                  validate="one_to_one", indicator=True)
    unmatched = paired["_merge"] != "both"
    if unmatched.any():
        counts = paired.loc[unmatched, "_merge"].value_counts().to_dict()
        examples = paired.loc[unmatched, PAIR_KEYS + ["_merge"]].head().to_dict("records")
        raise ValueError(f"Missing paired method results {counts}: {examples}")
    if not (paired["entry_date_left"] == paired["entry_date_right"]).all():
        raise ValueError("Entry dates disagree between paired methods")
    if not (paired["asset_id_left"] == paired["asset_id_right"]).all():
        raise ValueError("Asset IDs disagree between paired methods")

    paired = paired.drop(columns=["_merge", "entry_date_right", "asset_id_right"]).rename(
        columns={"entry_date_left": "entry_date", "asset_id_left": "asset_id"})
    paired["entry_date"] = pd.to_datetime(paired["entry_date"]).to_numpy(dtype="datetime64[D]")
    paired["diff"] = paired["score_left"] - paired["score_right"]
    paired["left_method"] = left
    paired["right_method"] = right
    return paired.sort_values(PAIR_KEYS, ignore_index=True)


def average_differences_across_seeds(paired, expected_seeds=None):
    """Average matched differences across seeds for each market/fold/window."""
    required = PAIR_KEYS + ["entry_date", "asset_id", "diff"]
    _require_columns(paired, required)
    if not np.isfinite(paired["diff"].to_numpy(dtype=float)).all():
        raise ValueError("Paired differences contain non-finite values")
    if paired.duplicated(PAIR_KEYS).any():
        raise ValueError("Paired differences contain duplicate market/fold/seed/window rows")

    expected = (tuple(sorted(int(seed) for seed in paired["seed"].unique()))
                if expected_seeds is None else tuple(sorted(int(seed) for seed in expected_seeds)))
    if not expected:
        raise ValueError("At least one expected seed is required")
    actual_global = tuple(sorted(int(seed) for seed in paired["seed"].unique()))
    if actual_global != expected:
        raise ValueError(f"Observed seeds {actual_global} do not match expected seeds {expected}")

    seed_sets = paired.groupby(WINDOW_KEYS, sort=False)["seed"].agg(
        lambda values: tuple(sorted(int(seed) for seed in values)))
    incomplete = seed_sets[seed_sets != expected]
    if not incomplete.empty:
        raise ValueError(f"{len(incomplete)} windows do not contain every expected seed")

    metadata_counts = paired.groupby(WINDOW_KEYS, sort=False)[["entry_date", "asset_id"]].nunique()
    if (metadata_counts > 1).any(axis=None):
        raise ValueError("Entry date or asset ID changes across seeds for the same window")

    averaged = paired.groupby(WINDOW_KEYS, as_index=False, sort=False).agg(
        entry_date=("entry_date", "first"),
        asset_id=("asset_id", "first"),
        diff=("diff", "mean"),
        n_seeds=("seed", "size"),
    )
    averaged["entry_date"] = pd.to_datetime(averaged["entry_date"]).to_numpy(
        dtype="datetime64[D]")
    return averaged


def paired_moving_block_bootstrap(paired, *, block_days=30, n_boot=10_000, seed=0,
                                  confidence=0.95, expected_seeds=None):
    """Estimate a paired mean, percentile interval and null-centered two-sided p-value.

    Seed differences are averaged per window first. Within each market/fold, moving
    blocks are spans of consecutive observed entry dates covering block_days calendar
    days. A sampled entry date always contributes every asset on that date. Each replicate
    samples each fold independently and pools its resampled windows using window weights.
    """
    if not isinstance(block_days, (int, np.integer)) or block_days < 1:
        raise ValueError("block_days must be a positive integer")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    averaged = average_differences_across_seeds(paired, expected_seeds)
    groups = _date_block_groups(averaged, int(block_days))
    rng = np.random.default_rng(seed)
    replicates = np.asarray([_resampled_mean(groups, rng) for _ in range(int(n_boot))])
    estimate = float(np.average(averaged["diff"]))
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicates, [alpha, 1.0 - alpha])
    null_replicates = replicates - estimate
    p_value = (1 + np.count_nonzero(np.abs(null_replicates) >= abs(estimate))) / (
        int(n_boot) + 1)

    return {
        "estimate": estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "p_value": float(p_value),
        "block_days": int(block_days),
        "n_boot": int(n_boot),
        "n_windows": int(len(averaged)),
        "n_dates": int(averaged.groupby(["market", "fold", "entry_date"]).ngroups),
        "n_folds": int(averaged.groupby(["market", "fold"]).ngroups),
        "n_markets": int(averaged["market"].nunique()),
        "n_seeds": int(averaged["n_seeds"].iloc[0]),
    }


def analyze_contrast(results, left, right, *, block_days=(15, 30, 60),
                     primary_block_days=30, n_boot=10_000, seed=0,
                     confidence=0.95, expected_seeds=None):
    """Run one contrast with block-length sensitivity and per-seed summaries."""
    lengths = tuple(dict.fromkeys(int(days) for days in block_days))
    if primary_block_days not in lengths:
        raise ValueError("primary_block_days must be included in block_days")
    paired = align_method_scores(results, left, right)
    sensitivity = [
        paired_moving_block_bootstrap(
            paired, block_days=days, n_boot=n_boot, seed=seed,
            confidence=confidence, expected_seeds=expected_seeds)
        for days in lengths
    ]
    primary = next(result for result in sensitivity
                   if result["block_days"] == primary_block_days)
    seed_rows = []
    for (market, seed_value), frame in paired.groupby(["market", "seed"], sort=True):
        seed_rows.append({
            "market": str(market),
            "seed": int(seed_value),
            "estimate": float(frame["diff"].mean()),
            "n_windows": int(len(frame)),
            "n_dates": int(frame.groupby(["fold", "entry_date"]).ngroups),
            "n_folds": int(frame["fold"].nunique()),
        })
    return {
        "left": left,
        "right": right,
        **primary,
        "per_seed": seed_rows,
        "block_length_sensitivity": sensitivity,
    }


def analyze_comparison_families(results, config, *, block_days=(15, 30, 60),
                                primary_block_days=30, n_boot=10_000, seed=0,
                                confidence=0.95):
    """Run config-defined primary and secondary families with Holm correction."""
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    expected_seeds = config.get("seeds")
    definitions = config["contrasts"]
    families = {
        "primary": [definitions["primary"]],
        "secondary": definitions.get("secondary", []),
    }
    output = {}
    for family, contrasts in families.items():
        rows = [
            analyze_contrast(
                results, contrast["left"], contrast["right"],
                block_days=block_days, primary_block_days=primary_block_days,
                n_boot=n_boot, seed=seed, confidence=confidence,
                expected_seeds=expected_seeds)
            for contrast in contrasts
        ]
        adjusted = holm_adjust([row["p_value"] for row in rows])
        for row, value in zip(rows, adjusted):
            row["p_value_holm"] = float(value)
            row["comparison_family"] = family
        output[family] = rows
    return output


def holm_adjust(p_values):
    """Holm step-down family-wise error adjustment in input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be a finite one-dimensional array in [0, 1]")
    if len(p) == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, ranked * np.arange(len(p), 0, -1)))
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def block_bootstrap_diff(diff, block_ids, n_boot=10_000, seed=0):
    """Compatibility wrapper returning mean, lower and upper."""
    diff = np.asarray(diff, dtype=float)
    block_ids = np.asarray(block_ids)
    if diff.ndim != 1 or block_ids.shape != diff.shape or not np.isfinite(diff).all():
        raise ValueError("diff and block_ids must be aligned finite one-dimensional arrays")
    unique = np.unique(block_ids)
    positions = {value: index for index, value in enumerate(unique)}
    dates = np.asarray([np.datetime64("2000-01-01") + np.timedelta64(positions[value], "D")
                        for value in block_ids])
    paired = pd.DataFrame({
        "market": "compat",
        "fold": 0,
        "seed": 0,
        "window_id": [f"compat:{index}" for index in range(len(diff))],
        "entry_date": dates,
        "asset_id": "compat",
        "diff": diff,
    })
    result = paired_moving_block_bootstrap(
        paired, block_days=1, n_boot=n_boot, seed=seed, expected_seeds=[0])
    return result["estimate"], result["ci_lower"], result["ci_upper"]


def _date_block_groups(averaged, block_days):
    groups = []
    for _, frame in averaged.groupby(["market", "fold"], sort=True):
        frame = frame.sort_values("entry_date")
        by_date = frame.groupby("entry_date", sort=True)["diff"].agg(["sum", "count"])
        dates = by_date.index.to_numpy(dtype="datetime64[D]")
        date_sums = by_date["sum"].to_numpy(dtype=float)
        date_counts = by_date["count"].to_numpy(dtype=int)
        if not len(dates):
            continue
        if dates[-1] - dates[0] < np.timedelta64(block_days, "D"):
            blocks = [np.arange(len(dates))]
        else:
            last_start = dates[-1] - np.timedelta64(block_days, "D")
            starts = list(np.flatnonzero(dates <= last_start))
            terminal = np.flatnonzero(dates > last_start)
            if len(terminal):
                starts.append(int(terminal[0]))
            blocks = [
                np.flatnonzero((dates >= dates[index]) &
                               (dates < dates[index] + np.timedelta64(block_days, "D")))
                for index in starts
            ]
        groups.append((date_sums, date_counts, blocks))
    if not groups:
        raise ValueError("No observations available for bootstrap")
    return groups


def _resampled_mean(groups, rng):
    total, count = 0.0, 0
    for date_sums, date_counts, blocks in groups:
        remaining = len(date_sums)
        while remaining:
            block = blocks[rng.integers(len(blocks))]
            chosen = block[:remaining]
            total += float(date_sums[chosen].sum())
            count += int(date_counts[chosen].sum())
            remaining -= len(chosen)
    return total / count


def _require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
