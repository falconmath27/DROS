"""Run and verify the P5 one-fold learned-method integration check.

Run from the repository root:

    python -m experiments.backtest.run

The default run uses the frozen S&P 100 inputs, fold 0, seed 0, both configured
objectives, and shortened OSPG training. Full research training is selected with
--full-training and belongs to the full-grid experiment stage.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import CostModel
from backtest.run import build_method, prepare
from common.problem import check_causal
from data.loaders import load_prices_csv
from data.splits import walk_forward_folds


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
METHODS = [
    "OSPG-GRU",
    "OSPG-MLP",
    "LSM",
    "Hold",
    "TakeProfitStopLoss",
    "TrailingStop",
]


def file_sha256(path):
    path = Path(path)
    return sha256(path.read_bytes()).hexdigest()


def verify_frozen_inputs(config, market):
    """Verify every configured P4 artifact before using the frozen market inputs."""
    market_config = config["markets"][market]
    verified = {}
    for name, artifact in market_config["inputs"].items():
        path = ROOT / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input: {path}")
        actual = file_sha256(path)
        if actual != artifact["sha256"]:
            raise ValueError(
                f"Frozen input hash mismatch for {market}/{name}: "
                f"expected {artifact['sha256']}, got {actual}")
        verified[name] = {"path": artifact["path"], "sha256": actual}
    return verified


def prepare_objective(config, market, objective, output_dir, cost_multiplier=1.0):
    """Rebuild one objective from the frozen P4 price and context inputs."""
    market_config = config["markets"][market]
    inputs = market_config["inputs"]
    prices = load_prices_csv(ROOT / inputs["clean_prices"]["path"])
    source_path = ROOT / inputs["source_manifest"]["path"]
    prices.attrs["manifest"] = json.loads(source_path.read_text())
    context = None
    if "context" in inputs:
        context = load_prices_csv(ROOT / inputs["context"]["path"])

    objective_config = next(
        item for item in config["objectives"] if item["name"] == objective)
    risk_aversion = float(objective_config.get("risk_aversion", 5.0))
    costs = CostModel(**market_config["costs"]).scaled(cost_multiplier)
    features = market_config["features"]
    horizon = config["horizon"]
    windows, manifest = prepare(
        prices,
        market=market,
        output_dir=output_dir,
        calendar_days=horizon["calendar_days"],
        sessions=horizon["sessions"],
        lookback=features["lookback_observations"],
        stride=features["window_stride_observations"],
        costs=costs,
        context=context,
        reward=objective,
        risk_aversion=risk_aversion,
        calendar=market_config["calendar"],
    )
    if context is not None:
        context.to_csv(output_dir / "context.csv")
    return windows, manifest, risk_aversion


def select_fold(windows, config, fold_number):
    split = config["walk_forward"]
    folds = walk_forward_folds(
        windows,
        train_months=split["train_months"],
        validation_months=split["validation_months"],
        test_months=split["test_months"],
        step_months=split["step_months"],
        embargo_days=split["additional_embargo_days"],
    )
    selected = next((fold for fold in folds if fold.number == fold_number), None)
    if selected is None:
        raise ValueError(
            f"Fold {fold_number} is unavailable; valid folds are "
            f"{[fold.number for fold in folds]}")
    return selected


def verify_deadline_purge(fold):
    validation_start = np.datetime64(fold.boundaries["validation_start"])
    test_start = np.datetime64(fold.boundaries["test_start"])
    checks = {
        "train_deadline_before_validation": bool(
            fold.train.metadata["deadline"].max() < validation_start),
        "validation_deadline_before_test": bool(
            fold.validation.metadata["deadline"].max() < test_start),
        "train_deadline_max": str(fold.train.metadata["deadline"].max()),
        "validation_start": str(validation_start),
        "validation_deadline_max": str(fold.validation.metadata["deadline"].max()),
        "test_start": str(test_start),
    }
    if not all(checks[name] for name in (
            "train_deadline_before_validation", "validation_deadline_before_test")):
        raise ValueError(f"Deadline purge failed: {checks}")
    return checks


def evaluate_one_fold(windows, fold, *, output_dir, seed, smoke, causal_paths):
    """Fit all six methods, save aligned arrays, and return verification records."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test = (
        part.problem for part in (fold.train, fold.validation, fold.test))
    arrays = {name: value.copy() for name, value in fold.test.metadata.items()}
    rows, method_checks = [], []

    for name in METHODS:
        model = build_method(name, seed=seed, smoke=smoke)
        model.fit(train.X, train.R, validation.X, validation.R)
        causal = bool(check_causal(
            model, test, n_paths=min(int(causal_paths), test.N), seed=seed))
        if not causal:
            raise ValueError(f"Prefix-causality check failed for {name}")

        raw_tau = np.asarray(model.stop(test.X, test.R), dtype=int)
        if raw_tau.shape != (test.N,) or (raw_tau < 0).any() or (raw_tau > test.H).any():
            raise ValueError(f"{name} returned invalid stopping indices")
        tau = np.minimum(raw_tau, fold.test.metadata["terminal_index"])
        reward = test.R[np.arange(test.N), tau]
        net_return = fold.test.metadata["net_returns"][np.arange(test.N), tau]
        exit_date = fold.test.metadata["decision_dates"][np.arange(test.N), tau]
        if not np.isfinite(reward).all() or not np.isfinite(net_return).all():
            raise ValueError(f"{name} produced non-finite test values")
        if (tau > fold.test.metadata["terminal_index"]).any():
            raise ValueError(f"{name} exits after a terminal index")
        if (exit_date > fold.test.metadata["end_date"]).any():
            raise ValueError(f"{name} exits after an observed terminal date")

        arrays[f"tau_{name}"] = tau.astype(np.int16)
        arrays[f"reward_{name}"] = reward
        arrays[f"objective_{name}"] = reward
        arrays[f"net_return_{name}"] = net_return
        arrays[f"exit_date_{name}"] = exit_date
        global_entry_decision = (
            bool(model.stop_at_0) if hasattr(model, "stop_at_0") else None)
        immediate_exit_rate = float(np.mean(tau == 0))
        row = {
            "fold": int(fold.number),
            "seed": int(seed),
            "method": name,
            "n_train": int(train.N),
            "n_validation": int(validation.N),
            "n_test": int(test.N),
            "mean_objective": float(reward.mean()),
            "mean_net_return": float(net_return.mean()),
            "mean_stop_index": float(tau.mean()),
            "immediate_exit_rate": immediate_exit_rate,
            "global_entry_decision": global_entry_decision,
            "prefix_causal": causal,
            "smoke": bool(smoke),
        }
        rows.append(row)
        method_checks.append({
            "method": name,
            "prefix_causal": causal,
            "finite_objective": bool(np.isfinite(reward).all()),
            "finite_net_return": bool(np.isfinite(net_return).all()),
            "exits_within_terminal_index": bool(
                (tau <= fold.test.metadata["terminal_index"]).all()),
            "exits_within_observed_end_date": bool(
                (exit_date <= fold.test.metadata["end_date"]).all()),
            "global_entry_decision": global_entry_decision,
            "immediate_exit_rate": immediate_exit_rate,
        })

    artifact = output_dir / f"fold_{fold.number}_seed_{seed}.npz"
    np.savez_compressed(artifact, **arrays)
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    return artifact, method_checks


def verify_saved_artifact(path):
    """Check method alignment and required saved fields after reopening the NPZ."""
    with np.load(path, allow_pickle=False) as data:
        n = len(data["window_id"])
        if len(np.unique(data["window_id"])) != n:
            raise ValueError("Saved fold contains duplicate window IDs")
        for method in METHODS:
            for prefix in ("tau", "reward", "objective", "net_return", "exit_date"):
                key = f"{prefix}_{method}"
                if key not in data.files or data[key].shape != (n,):
                    raise ValueError(f"Saved fold is missing aligned array {key}")
            if not np.isfinite(data[f"reward_{method}"]).all():
                raise ValueError(f"Saved reward_{method} is non-finite")
            if not np.isfinite(data[f"net_return_{method}"]).all():
                raise ValueError(f"Saved net_return_{method} is non-finite")
            if (data[f"tau_{method}"] > data["terminal_index"]).any():
                raise ValueError(f"Saved tau_{method} exceeds terminal_index")
            if (data[f"exit_date_{method}"] > data["end_date"]).any():
                raise ValueError(f"Saved exit_date_{method} exceeds end_date")
        return {
            "n_windows": int(n),
            "window_id_sha256": _array_sha256(data["window_id"]),
            "entry_date_sha256": _array_sha256(data["entry_date"]),
            "asset_id_sha256": _array_sha256(data["asset_id"]),
        }


def run_objective(config, *, config_path, market, objective, fold_number, seed,
                  output_root, smoke, causal_paths, verified_inputs):
    output_dir = Path(output_root) / market / objective
    output_dir.mkdir(parents=True, exist_ok=True)
    windows, manifest, risk_aversion = prepare_objective(
        config, market, objective, output_dir)
    fold = select_fold(windows, config, fold_number)
    purge_checks = verify_deadline_purge(fold)
    artifact, method_checks = evaluate_one_fold(
        windows, fold, output_dir=output_dir, seed=seed, smoke=smoke,
        causal_paths=causal_paths)
    artifact_checks = verify_saved_artifact(artifact)
    feature_sha256 = _array_sha256(fold.test.problem.X)

    resolved = {
        "task": "P5 one-fold learned run",
        "source_config": str(Path(config_path).resolve()),
        "source_config_sha256": file_sha256(config_path),
        "market": market,
        "objective": objective,
        "risk_aversion": risk_aversion if objective == "exponential" else None,
        "fold": int(fold_number),
        "seed": int(seed),
        "methods": METHODS,
        "smoke": bool(smoke),
        "training_budget": (
            {"OSPG": {"steps": 20, "min_steps": 0, "eval_every": 10,
                      "batch_norm": False}, "LSM": "default", "rules": "fixed"}
            if smoke else {"OSPG": "default full budget", "LSM": "default",
                           "rules": "fixed"}),
        "verified_inputs": verified_inputs,
        "generated_windows_sha256": file_sha256(output_dir / "windows.npz"),
        "test_features_sha256": feature_sha256,
        "test_window_id_sha256": artifact_checks["window_id_sha256"],
        "fold_boundaries": fold.boundaries,
        "dataset": {
            "assets": len(manifest["assets"]),
            "windows": int(windows.problem.N),
            "train_windows": int(fold.train.problem.N),
            "validation_windows": int(fold.validation.problem.N),
            "test_windows": int(fold.test.problem.N),
        },
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n")
    verification = {
        "passed": True,
        "deadline_purge": purge_checks,
        "artifact": artifact_checks,
        "methods": method_checks,
    }
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n")
    return {
        "objective": objective,
        "output": str(output_dir),
        "artifact": str(artifact),
        "test_features_sha256": feature_sha256,
        **artifact_checks,
    }


def compare_objectives(results):
    """Require both objectives to use the identical test feature and metadata rows."""
    if len(results) < 2:
        return {"passed": True, "objectives": [row["objective"] for row in results]}
    reference = results[0]
    fields = [
        "n_windows",
        "window_id_sha256",
        "entry_date_sha256",
        "asset_id_sha256",
        "test_features_sha256",
    ]
    mismatches = {
        row["objective"]: [field for field in fields if row[field] != reference[field]]
        for row in results[1:]
    }
    mismatches = {name: fields for name, fields in mismatches.items() if fields}
    if mismatches:
        raise ValueError(f"Objectives do not share identical test windows: {mismatches}")
    return {
        "passed": True,
        "reference_objective": reference["objective"],
        "objectives": [row["objective"] for row in results],
        "matched_fields": fields,
        "n_windows": reference["n_windows"],
    }


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--market", choices=["nifty50", "sp100", "crypto"],
                        default="sp100")
    parser.add_argument("--objectives", nargs="+", choices=["linear", "exponential"],
                        default=["linear", "exponential"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--causal-paths", type=int, default=100)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results" / "p5" / "one_fold")
    parser.add_argument("--full-training", action="store_true")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.market not in config["markets"]:
        raise ValueError(f"Market {args.market} is absent from the config")
    if args.seed not in config["seeds"]:
        raise ValueError(f"Seed {args.seed} is absent from the frozen config")
    configured_objectives = {item["name"] for item in config["objectives"]}
    if not set(args.objectives) <= configured_objectives:
        raise ValueError("Requested objective is absent from the frozen config")
    if args.fold < 0 or args.causal_paths < 1:
        raise ValueError("fold must be nonnegative and causal-paths must be positive")

    verified_inputs = verify_frozen_inputs(config, args.market)
    results = [
        run_objective(
            config,
            config_path=args.config,
            market=args.market,
            objective=objective,
            fold_number=args.fold,
            seed=args.seed,
            output_root=args.output_root,
            smoke=not args.full_training,
            causal_paths=args.causal_paths,
            verified_inputs=verified_inputs,
        )
        for objective in args.objectives
    ]
    objective_check = compare_objectives(results)
    run_summary = {
        "passed": True,
        "market": args.market,
        "fold": args.fold,
        "seed": args.seed,
        "smoke": not args.full_training,
        "objective_alignment": objective_check,
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"{args.market}_fold_{args.fold}_seed_{args.seed}.json"
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2))
    return run_summary


def _array_sha256(array):
    array = np.asarray(array)
    digest = sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    if array.dtype.kind in {"U", "O"}:
        digest.update("\n".join(array.astype(str).ravel()).encode())
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
