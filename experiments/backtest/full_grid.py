"""Resumable P5 full-grid and cost-sensitivity runner.

The default command runs every configured market, objective, cost multiplier, fold,
seed and method with full training budgets:

    python -m experiments.backtest.full_grid

Each method is checkpointed as an aligned fold artifact or a recorded failure. Use
--smoke only to validate orchestration; smoke outputs are isolated from final results.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from backtest.run import build_method
from common.problem import check_causal
from data.splits import split_at
from data.windows import MarketWindows
from experiments.backtest.run import (
    DEFAULT_CONFIG,
    METHODS,
    ROOT,
    _array_sha256,
    file_sha256,
    prepare_objective,
    verify_deadline_purge,
    verify_frozen_inputs,
)

_WORKER_DATA = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cost_label(multiplier):
    value = float(multiplier)
    if value < 0 or not np.isfinite(value):
        raise ValueError("Cost multipliers must be finite and nonnegative")
    return f"cost_{value:g}x"


def scenario_id(config_path, market, objective, cost_multiplier, smoke):
    payload = {
        "config_sha256": file_sha256(config_path),
        "market": market,
        "objective": objective,
        "cost_multiplier": float(cost_multiplier),
        "smoke": bool(smoke),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_scenario(config, *, config_path, market, objective, cost_multiplier,
                     output_dir, smoke, verified_inputs):
    """Prepare or resume one market/objective/cost dataset."""
    expected_id = scenario_id(
        config_path, market, objective, cost_multiplier, smoke)
    resolved_path = output_dir / "resolved_config.json"
    windows_path = output_dir / "windows.npz"
    failure_path = output_dir / "scenario_failure.json"
    if resolved_path.is_file() and windows_path.is_file():
        resolved = json.loads(resolved_path.read_text())
        if (resolved.get("scenario_id") == expected_id and
                resolved.get("windows_sha256") == file_sha256(windows_path)):
            failure_path.unlink(missing_ok=True)
            return MarketWindows.load(windows_path), resolved

    output_dir.mkdir(parents=True, exist_ok=True)
    windows, manifest, risk_aversion = prepare_objective(
        config, market, objective, output_dir,
        cost_multiplier=float(cost_multiplier))
    resolved = {
        "scenario_id": expected_id,
        "created_at": utc_now(),
        "source_config": str(Path(config_path).resolve()),
        "source_config_sha256": file_sha256(config_path),
        "market": market,
        "objective": objective,
        "risk_aversion": risk_aversion if objective == "exponential" else None,
        "cost_multiplier": float(cost_multiplier),
        "base_costs": config["markets"][market]["costs"],
        "cost_policy": (
            "Rewards are rebuilt and every learned policy is refit for this scenario. "
            "No fixed-exit rescoring is reported."),
        "methods": METHODS,
        "smoke": bool(smoke),
        "training_budget": (
            {"OSPG": {"steps": 20, "min_steps": 0, "eval_every": 10,
                      "batch_norm": False}, "LSM": "default", "rules": "fixed"}
            if smoke else {"OSPG": "default full budget", "LSM": "default",
                           "rules": "fixed"}),
        "verified_inputs": verified_inputs,
        "windows_sha256": file_sha256(windows_path),
        "dataset": {
            "assets": len(manifest["assets"]),
            "windows": int(windows.problem.N),
            "features": list(windows.feature_names),
        },
        "limitations": [
            "The equity constituents and crypto ranking are fixed supplied or "
            "retrieval-date universes, not historical point-in-time membership.",
            "Entry and exit use a stylized adjusted daily-close reference price with "
            "assumed friction; same-close execution is not guaranteed.",
            "Overlapping windows are not a self-financing portfolio return.",
        ],
    }
    _write_json(resolved_path, resolved)
    failure_path.unlink(missing_ok=True)
    return windows, resolved


def configured_fold(windows, config, fold_number):
    """Materialize one requested fold without duplicating every expanding train set."""
    split = config["walk_forward"]
    origin = pd.Timestamp(windows.metadata["entry_date"].min())
    validation_start = origin + pd.DateOffset(
        months=split["train_months"] + fold_number * split["step_months"])
    test_start = validation_start + pd.DateOffset(
        months=split["validation_months"])
    test_end = test_start + pd.DateOffset(months=split["test_months"])
    return split_at(
        windows,
        validation_start,
        test_start,
        test_end,
        embargo_days=split["additional_embargo_days"],
        number=fold_number,
    )


def fold_summary(windows, config, fold_number):
    fold = configured_fold(windows, config, fold_number)
    return {
        "number": fold.number,
        "n_train": fold.train.problem.N,
        "n_validation": fold.validation.problem.N,
        "n_test": fold.test.problem.N,
        **fold.boundaries,
    }


def run_fold_seed(windows, fold, *, output_dir, seed, methods, smoke,
                  causal_paths, force=False, write_metrics=True):
    """Run methods independently and checkpoint after each completion or failure."""
    output_dir = Path(output_dir)
    artifact = output_dir / f"fold_{fold.number}_seed_{seed}.npz"
    status_path = output_dir / f"fold_{fold.number}_seed_{seed}.status.json"
    expected_metadata = fold.test.metadata
    arrays = {name: value.copy() for name, value in expected_metadata.items()}
    status = {
        "market": str(expected_metadata["market"][0]),
        "fold": int(fold.number),
        "seed": int(seed),
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "deadline_purge": verify_deadline_purge(fold),
        "n_train": int(fold.train.problem.N),
        "n_validation": int(fold.validation.problem.N),
        "n_test": int(fold.test.problem.N),
        "window_id_sha256": _array_sha256(expected_metadata["window_id"]),
        "entry_date_sha256": _array_sha256(expected_metadata["entry_date"]),
        "asset_id_sha256": _array_sha256(expected_metadata["asset_id"]),
        "test_features_sha256": _array_sha256(fold.test.problem.X),
        "methods": {},
    }
    if artifact.is_file() and status_path.is_file() and not force:
        saved_status = json.loads(status_path.read_text())
        with np.load(artifact, allow_pickle=False) as saved:
            if _metadata_matches(saved, expected_metadata):
                arrays.update({name: saved[name] for name in saved.files})
                status.update(saved_status)

    train, validation, test = (
        part.problem for part in (fold.train, fold.validation, fold.test))
    for method in methods:
        previous = status["methods"].get(method, {})
        if (not force and previous.get("status") == "completed" and
                _method_arrays_valid(arrays, method, expected_metadata)):
            continue
        method_status = {
            "status": "running",
            "started_at": utc_now(),
            "completed_at": None,
            "error": None,
        }
        status["methods"][method] = method_status
        status["updated_at"] = utc_now()
        _write_json(status_path, status)
        try:
            model = build_method(method, seed=seed, smoke=smoke)
            model.fit(train.X, train.R, validation.X, validation.R)
            causal = bool(check_causal(
                model, test, n_paths=min(causal_paths, test.N), seed=seed))
            if not causal:
                raise ValueError("Prefix-causality check failed")
            raw_tau = np.asarray(model.stop(test.X, test.R), dtype=int)
            if (raw_tau.shape != (test.N,) or (raw_tau < 0).any() or
                    (raw_tau > test.H).any()):
                raise ValueError("Method returned invalid stopping indices")
            tau = np.minimum(raw_tau, expected_metadata["terminal_index"])
            objective = test.R[np.arange(test.N), tau]
            net_return = expected_metadata["net_returns"][np.arange(test.N), tau]
            exit_date = expected_metadata["decision_dates"][np.arange(test.N), tau]
            if not np.isfinite(objective).all() or not np.isfinite(net_return).all():
                raise ValueError("Method produced non-finite results")
            if (tau > expected_metadata["terminal_index"]).any():
                raise ValueError("Method exits after terminal_index")
            if (exit_date > expected_metadata["end_date"]).any():
                raise ValueError("Method exits after end_date")

            arrays[f"tau_{method}"] = tau.astype(np.int16)
            arrays[f"reward_{method}"] = objective
            arrays[f"objective_{method}"] = objective
            arrays[f"net_return_{method}"] = net_return
            arrays[f"exit_date_{method}"] = exit_date
            _write_npz(artifact, arrays)
            method_status.update({
                "status": "completed",
                "completed_at": utc_now(),
                "prefix_causal": causal,
                "mean_objective": float(objective.mean()),
                "mean_net_return": float(net_return.mean()),
                "mean_stop_index": float(tau.mean()),
                "immediate_exit_rate": float(np.mean(tau == 0)),
                "global_entry_decision": (
                    bool(model.stop_at_0) if hasattr(model, "stop_at_0") else None),
            })
        except Exception as error:
            method_status.update({
                "status": "failed",
                "completed_at": utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            })
        status["updated_at"] = utc_now()
        _write_json(status_path, status)

    statuses = [status["methods"].get(method, {}).get("status") for method in methods]
    status["status"] = "completed" if all(value == "completed" for value in statuses) else "failed"
    status["updated_at"] = utc_now()
    _write_json(status_path, status)
    if write_metrics:
        _write_metrics(output_dir)
    return status


def run_scenario_jobs(windows, config, fold_numbers, *, output_dir, seeds,
                      methods, smoke, causal_paths, force, workers,
                      progress_interval=10.0, progress_batch_size=3):
    """Run fold/seed jobs with periodic heartbeats and batched progress reports."""
    jobs = [(fold_number, seed) for fold_number in fold_numbers for seed in seeds]
    started = time.monotonic()
    print(
        f"[P5] Queued {len(jobs)} fold/seed jobs x {len(methods)} methods "
        f"with {workers} worker(s). Progress updates every "
        f"{progress_interval:g}s.",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _worker_fold_seed,
                str(Path(output_dir) / "windows.npz"),
                config,
                fold_number,
                seed,
                str(output_dir),
                methods,
                smoke,
                causal_paths,
                force,
            ): (fold_number, seed)
            for fold_number, seed in jobs
        }
        pending = set(futures)
        completed_jobs = 0
        reported_jobs = 0
        batch_number = 0
        while pending:
            done, pending = wait(
                pending,
                timeout=progress_interval,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                future.result()
                completed_jobs += 1

            should_report_batch = (
                completed_jobs - reported_jobs >= progress_batch_size or
                (done and not pending)
            )
            snapshot = _progress_snapshot(output_dir, jobs, methods)
            elapsed = time.monotonic() - started
            if should_report_batch:
                batch_number += 1
                reported_jobs = completed_jobs
                label = f"Batch {batch_number}"
            elif not done:
                label = "Heartbeat"
            else:
                continue
            print(_progress_line(
                label,
                completed_jobs=completed_jobs,
                total_jobs=len(jobs),
                workers=workers,
                elapsed=elapsed,
                snapshot=snapshot,
            ), flush=True)
    _write_metrics(output_dir)


def _progress_snapshot(output_dir, jobs, methods):
    """Read lightweight status checkpoints without opening large artifacts."""
    counts = {state: 0 for state in ("completed", "failed", "running", "pending")}
    active = []
    output_dir = Path(output_dir)
    for fold_number, seed in jobs:
        status_path = output_dir / f"fold_{fold_number}_seed_{seed}.status.json"
        status = None
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                status = None
        method_statuses = status.get("methods", {}) if status else {}
        current = []
        for method in methods:
            state = method_statuses.get(method, {}).get("status", "pending")
            if state not in counts:
                state = "pending"
            counts[state] += 1
            if state == "running":
                current.append(method)
        if current:
            active.append(
                f"fold {fold_number}/seed {seed}: {','.join(current)}")
    return {"counts": counts, "active": active}


def _progress_line(label, *, completed_jobs, total_jobs, workers, elapsed,
                   snapshot):
    counts = snapshot["counts"]
    fraction = completed_jobs / total_jobs if total_jobs else 1.0
    if completed_jobs:
        remaining_jobs = max(0, total_jobs - completed_jobs)
        eta = elapsed * remaining_jobs / completed_jobs
        eta_text = _format_duration(eta)
    else:
        eta_text = "calculating"
    active = snapshot["active"]
    active_text = "; ".join(active[:6]) if active else "workers starting/checkpointing"
    return (
        f"[P5] {label} | jobs {completed_jobs}/{total_jobs} "
        f"({fraction:.0%}) | methods {counts['completed']} complete, "
        f"{counts['failed']} failed, {counts['running']} running | "
        f"elapsed {_format_duration(elapsed)} | ETA {eta_text} | "
        f"active: {active_text}"
    )


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    return f"{minutes:d}m {seconds:02d}s"


def _worker_fold_seed(windows_path, config, fold_number, seed, output_dir,
                      methods, smoke, causal_paths, force):
    _set_worker_threads()
    cache_key = str(Path(windows_path).resolve())
    if cache_key not in _WORKER_DATA:
        windows = MarketWindows.load(windows_path)
        _WORKER_DATA[cache_key] = windows
    windows = _WORKER_DATA[cache_key]
    fold = configured_fold(windows, config, fold_number)
    return run_fold_seed(
        windows,
        fold,
        output_dir=output_dir,
        seed=seed,
        methods=methods,
        smoke=smoke,
        causal_paths=causal_paths,
        force=force,
        write_metrics=False,
    )


def _set_worker_threads():
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def record_scenario_failure(output_dir, *, market, objective, cost_multiplier,
                            folds, seeds, methods, error):
    output_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "status": "failed",
        "scope": "scenario_preparation",
        "market": market,
        "objective": objective,
        "cost_multiplier": float(cost_multiplier),
        "affected_folds": list(folds),
        "affected_seeds": list(seeds),
        "affected_methods": list(methods),
        "recorded_at": utc_now(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    }
    _write_json(output_dir / "scenario_failure.json", failure)


def collect_coverage(output_root, scope):
    """Report every selected method run as completed, failed or pending."""
    records = []
    support = {}
    for market in scope["markets"]:
        for objective in scope["objectives"]:
            for multiplier in scope["cost_multipliers"]:
                scenario = Path(output_root) / market / objective / cost_label(multiplier)
                scenario_failure = scenario / "scenario_failure.json"
                if scenario_failure.is_file():
                    failure = json.loads(scenario_failure.read_text())
                    for fold in scope["folds_by_market"][market]:
                        for seed in scope["seeds"]:
                            for method in scope["methods"]:
                                records.append(_coverage_row(
                                    market, objective, multiplier, fold, seed, method,
                                    "failed", None, failure["error"]))
                    continue
                for fold in scope["folds_by_market"][market]:
                    for seed in scope["seeds"]:
                        status_path = scenario / f"fold_{fold}_seed_{seed}.status.json"
                        artifact = scenario / f"fold_{fold}_seed_{seed}.npz"
                        status = (json.loads(status_path.read_text())
                                  if status_path.is_file() else None)
                        for method in scope["methods"]:
                            method_status = (
                                status.get("methods", {}).get(method, {})
                                if status else {})
                            state = method_status.get("status", "pending")
                            valid = (
                                state == "completed" and artifact.is_file() and
                                _saved_method_valid(artifact, method))
                            if state == "completed" and not valid:
                                state = "failed"
                                method_status = {
                                    "error": {
                                        "type": "ArtifactValidationError",
                                        "message": "Completed status lacks a valid artifact",
                                    }}
                            records.append(_coverage_row(
                                market, objective, multiplier, fold, seed, method,
                                state, str(artifact) if artifact.is_file() else None,
                                method_status.get("error")))
                        if status:
                            key = f"{market}:fold_{fold}"
                            fingerprint = (
                                status.get("window_id_sha256"),
                                status.get("entry_date_sha256"),
                                status.get("asset_id_sha256"),
                                status.get("test_features_sha256"),
                            )
                            if all(fingerprint):
                                support.setdefault(key, set()).add(fingerprint)

    mismatches = {
        key: len(fingerprints) for key, fingerprints in support.items()
        if len(fingerprints) != 1
    }
    counts = {
        state: sum(record["status"] == state for record in records)
        for state in ("completed", "failed", "pending")
    }
    coverage = {
        "generated_at": utc_now(),
        "scope": scope,
        "counts": {"expected": len(records), **counts},
        "matching_support": {
            "passed": not mismatches,
            "checked_market_folds": len(support),
            "mismatches": mismatches,
        },
        "passed": (
            counts["completed"] == len(records) and
            counts["failed"] == 0 and counts["pending"] == 0 and not mismatches),
        "limitations": [
            "Current/retrieval-date market universes introduce survivorship and "
            "selection bias.",
            "Daily-close reference execution is stylized and does not guarantee a "
            "same-close fill.",
            "FX and gold are optional and are not part of the configured core grid.",
        ],
        "runs": records,
    }
    _write_json(Path(output_root) / "coverage.json", coverage)
    pd.DataFrame(records).to_csv(Path(output_root) / "coverage.csv", index=False)
    return coverage


def _coverage_row(market, objective, multiplier, fold, seed, method,
                  status, artifact, error):
    return {
        "market": market,
        "objective": objective,
        "cost_multiplier": float(multiplier),
        "fold": int(fold),
        "seed": int(seed),
        "method": method,
        "status": status,
        "artifact": artifact,
        "error_type": error.get("type") if error else None,
        "error_message": error.get("message") if error else None,
    }


def _metadata_matches(saved, expected):
    return all(
        name in saved.files and np.array_equal(saved[name], value)
        for name, value in expected.items())


def _method_arrays_valid(arrays, method, metadata):
    keys = [
        f"tau_{method}",
        f"reward_{method}",
        f"objective_{method}",
        f"net_return_{method}",
        f"exit_date_{method}",
    ]
    if any(key not in arrays or arrays[key].shape != (len(metadata["window_id"]),)
           for key in keys):
        return False
    return (
        np.isfinite(arrays[f"objective_{method}"]).all() and
        np.isfinite(arrays[f"net_return_{method}"]).all() and
        (arrays[f"tau_{method}"] <= metadata["terminal_index"]).all() and
        (arrays[f"exit_date_{method}"] <= metadata["end_date"]).all())


def _saved_method_valid(artifact, method):
    try:
        with np.load(artifact, allow_pickle=False) as data:
            metadata = {name: data[name] for name in (
                "window_id", "terminal_index", "end_date")}
            arrays = {name: data[name] for name in data.files}
            return _method_arrays_valid(arrays, method, metadata)
    except Exception:
        return False


def _write_metrics(output_dir):
    rows = []
    for path in sorted(Path(output_dir).glob("fold_*_seed_*.status.json")):
        status = json.loads(path.read_text())
        for method, values in status.get("methods", {}).items():
            if values.get("status") == "completed":
                rows.append({
                    "fold": status["fold"],
                    "seed": status["seed"],
                    "method": method,
                    "n_test": status["n_test"],
                    "mean_objective": values["mean_objective"],
                    "mean_net_return": values["mean_net_return"],
                    "mean_stop_index": values["mean_stop_index"],
                    "immediate_exit_rate": values["immediate_exit_rate"],
                    "global_entry_decision": values["global_entry_decision"],
                    "prefix_causal": values["prefix_causal"],
                })
    pd.DataFrame(rows).to_csv(Path(output_dir) / "metrics.csv", index=False)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    _replace_with_retry(temporary, path)


def _write_npz(path, arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    _replace_with_retry(temporary, path)


def _replace_with_retry(temporary, destination, attempts=8):
    """Tolerate short OneDrive locks while atomically replacing a checkpoint."""
    for attempt in range(attempts):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * 2 ** attempt)


def _write_result_readme(output_root, smoke):
    title = "P5 grid smoke validation" if smoke else "P5 full-grid results"
    text = f"""# {title}

Each scenario rebuilds its reward and refits OSPG-GRU, OSPG-MLP and LSM. The three
rule methods are evaluated on the same windows. Cost scenarios are 0x, 1x and 2x the
configured venue assumptions. Fixed-exit rescoring is not mixed into these results.

The equity constituents and crypto ranking are fixed supplied or retrieval-date
universes. They are not historical point-in-time membership and therefore introduce
survivorship and selection bias.

Entry and exit use adjusted daily-close reference prices plus assumed friction. This
is a stylized execution model and does not claim that an order placed after observing
the close fills at that close.

FX and gold remain optional and are outside the configured core grid.
"""
    (Path(output_root) / "README.md").write_text(text)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--markets", nargs="+",
                        choices=["nifty50", "sp100", "crypto"])
    parser.add_argument("--objectives", nargs="+",
                        choices=["linear", "exponential"])
    parser.add_argument("--cost-multipliers", nargs="+", type=float,
                        default=[0.0, 1.0, 2.0])
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--causal-paths", type=int, default=100)
    parser.add_argument("--workers", type=int,
                        default=min(6, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument(
        "--progress-interval", type=float, default=10.0,
        help="Seconds between heartbeat messages while workers are busy.")
    parser.add_argument(
        "--progress-batch-size", type=int, default=3,
        help="Completed fold/seed jobs grouped into each batch summary.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    config = json.loads(args.config.read_text())
    markets = args.markets or list(config["markets"])
    objectives = args.objectives or [item["name"] for item in config["objectives"]]
    seeds = args.seeds or list(config["seeds"])
    methods = args.methods or list(config["methods"])
    if not set(seeds) <= set(config["seeds"]):
        raise ValueError("Every seed must be present in the frozen config")
    if not set(methods) <= set(config["methods"]):
        raise ValueError("Every method must be present in the frozen config")
    if (args.causal_paths < 1 or args.workers < 1 or
            args.progress_interval <= 0 or args.progress_batch_size < 1):
        raise ValueError(
            "causal-paths, workers, progress-interval and "
            "progress-batch-size must be positive")
    for multiplier in args.cost_multipliers:
        cost_label(multiplier)
    output_root = args.output_root or (
        ROOT / "results" / "p5" /
        ("grid_smoke" if args.smoke else "full_grid"))
    output_root.mkdir(parents=True, exist_ok=True)
    _write_result_readme(output_root, args.smoke)

    folds_by_market = {}
    for market in markets:
        available = int(config["markets"][market]["dataset"]["folds"])
        selected = list(range(available)) if args.folds is None else list(args.folds)
        invalid = set(selected) - set(range(available))
        if invalid:
            raise ValueError(f"Invalid folds for {market}: {sorted(invalid)}")
        folds_by_market[market] = selected
    scope = {
        "markets": markets,
        "objectives": objectives,
        "cost_multipliers": [float(value) for value in args.cost_multipliers],
        "folds_by_market": folds_by_market,
        "seeds": seeds,
        "methods": methods,
        "smoke": bool(args.smoke),
        "workers": int(args.workers),
    }
    _write_json(output_root / "scope.json", scope)

    total_scenarios = (
        len(markets) * len(objectives) * len(args.cost_multipliers))
    scenario_number = 0
    print(
        f"[P5] Starting {total_scenarios} scenario(s): "
        f"{', '.join(markets)} | {', '.join(objectives)} | "
        f"costs {', '.join(cost_label(value) for value in args.cost_multipliers)}",
        flush=True,
    )

    for market in markets:
        verified_inputs = verify_frozen_inputs(config, market)
        for objective in objectives:
            for multiplier in args.cost_multipliers:
                scenario_number += 1
                scenario_dir = (
                    output_root / market / objective / cost_label(multiplier))
                print(
                    f"\n[P5] Scenario {scenario_number}/{total_scenarios}: "
                    f"{market} | {objective} | {cost_label(multiplier)}",
                    flush=True,
                )
                try:
                    print(
                        "[P5] Preparing or loading frozen scenario data...",
                        flush=True,
                    )
                    windows, resolved = prepare_scenario(
                        config,
                        config_path=args.config,
                        market=market,
                        objective=objective,
                        cost_multiplier=multiplier,
                        output_dir=scenario_dir,
                        smoke=args.smoke,
                        verified_inputs=verified_inputs,
                    )
                    print(
                        f"[P5] Data ready: {windows.problem.N:,} windows, "
                        f"{resolved['dataset']['assets']} assets.",
                        flush=True,
                    )
                    _write_json(scenario_dir / "folds.json", [
                        fold_summary(windows, config, fold_number)
                        for fold_number in folds_by_market[market]
                    ])
                    run_scenario_jobs(
                        windows,
                        config,
                        folds_by_market[market],
                        output_dir=scenario_dir,
                        seeds=seeds,
                        methods=methods,
                        smoke=args.smoke,
                        causal_paths=args.causal_paths,
                        force=args.force,
                        workers=args.workers,
                        progress_interval=args.progress_interval,
                        progress_batch_size=args.progress_batch_size,
                    )
                except Exception as error:
                    record_scenario_failure(
                        scenario_dir,
                        market=market,
                        objective=objective,
                        cost_multiplier=multiplier,
                        folds=folds_by_market[market],
                        seeds=seeds,
                        methods=methods,
                        error=error,
                    )
                coverage = collect_coverage(output_root, scope)
                print(json.dumps(coverage["counts"]), flush=True)

    coverage = collect_coverage(output_root, scope)
    print(json.dumps({
        "passed": coverage["passed"],
        "counts": coverage["counts"],
        "matching_support": coverage["matching_support"],
        "coverage": str(output_root / "coverage.json"),
    }, indent=2))
    if not coverage["passed"]:
        raise RuntimeError(
            "Grid finished with failed or pending runs; inspect coverage.json")
    return coverage


if __name__ == "__main__":
    main()
