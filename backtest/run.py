"""P4 end-to-end harness. Run python -m backtest.run --help from the repo root."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

from backtest.costs import CostModel, venue_costs
from backtest.rules import HoldToEnd, TakeProfitStopLoss, TrailingStop
from data.clean import clean_prices
from data.loaders import CACHE, load_crypto, load_equities, load_prices_csv
from data.splits import walk_forward_folds
from data.windows import make_windows

ROOT = Path(__file__).resolve().parents[1]


def prepare(prices, *, market, output_dir, calendar_days=30, sessions=None,
            lookback=20, stride=5, costs=None, context=None, reward="linear",
            risk_aversion=5.0, min_observations=60, calendar=None):
    """Save dataset, source manifest and cleaning log. Same inputs give stable window IDs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crypto = calendar == "crypto" if calendar is not None else market == "crypto"
    clean, audit = clean_prices(prices, calendar="crypto" if crypto else "equity",
                               min_observations=min_observations)
    windows = make_windows(clean, market=market, calendar_days=calendar_days, H=sessions,
        lookback=lookback, stride=stride, costs=costs, context=context,
        annualization=365 if crypto else 252, reward=reward, risk_aversion=risk_aversion)
    source = prices.attrs.get("manifest", {})
    clean.to_csv(output_dir / "clean_prices.csv")
    (output_dir / "cleaning_log.json").write_text(json.dumps(audit, indent=2))
    (output_dir / "source_manifest.json").write_text(json.dumps(source, indent=2))
    windows.save(output_dir / "windows.npz")
    pd.DataFrame({k: v for k, v in windows.metadata.items() if v.ndim == 1}).to_csv(
        output_dir / "windows.csv", index=False)
    manifest = dict(market=market, assets=list(clean.columns), observations=len(clean),
        first_date=str(clean.index.min().date()), last_date=str(clean.index.max().date()),
        n_windows=windows.problem.N, features=windows.feature_names, config=windows.config,
        source=source, limitations=["Fixed/retrieval-date universe is not point-in-time membership.",
        "Daily-close reference execution; same-close fills are not guaranteed.",
        "Cost scenarios are assumptions, not official historical tariffs.",
        "Delisting returns unavailable; no invented terminal payouts.",
        "Overlapping windows require P5 date-block bootstrap; no portfolio CAGR inferred."])
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return windows, manifest


def build_method(name, seed=0, smoke=False):
    rules = {"Hold": HoldToEnd, "TakeProfitStopLoss": TakeProfitStopLoss, "TrailingStop": TrailingStop}
    if name in rules:
        return rules[name]()
    # Lazy imports: data preparation and rules do not require PyTorch.
    if name == "LSM":
        from baselines.lsm import LSM
        return LSM(itm_only=False)
    if name == "DOS":
        from baselines.dos import DOS
        return DOS(itm_only=False, seed=seed, **(dict(steps=3) if smoke else {}))
    if name.startswith("OSPG-"):
        from ospg.train import OSPG
        return OSPG(kind=name.split("-")[1].lower(), itm_only=False, seed=seed,
            **(dict(steps=20, min_steps=0, eval_every=10, batch_norm=False) if smoke else {}))
    if name.startswith("FQI-"):
        from baselines.fqi import FQI
        return FQI(kind=name.split("-")[1].lower(), itm_only=False, seed=seed,
            **(dict(steps=20, eval_every=10) if smoke else {}))
    raise ValueError(f"Unknown method {name}")


def evaluate_folds(windows, folds, methods, *, output_dir, seeds=(0,), smoke=False):
    """Isolated P4 results; preserve P1/P2 raw.jsonl and existing synthetic experiments.

    Existing models assume common date-zero states; market entry states differ. Their
    current global date-zero rule is retained and disclosed, not silently redesigned.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata_keys = tuple(windows.metadata)
    for fold in folds:
        tr, va, te = (part.problem for part in (fold.train, fold.validation, fold.test))
        for seed in seeds:
            arrays = {k: fold.test.metadata[k] for k in metadata_keys}
            for name in methods:
                model = build_method(name, seed, smoke)
                model.fit(tr.X, tr.R, va.X, va.R)
                tau = np.asarray(model.stop(te.X, te.R), dtype=int)
                if tau.shape != (te.N,) or (tau < 0).any() or (tau > te.H).any():
                    raise ValueError(f"{name} returned invalid stopping indices")
                tau = np.minimum(tau, fold.test.metadata["terminal_index"])
                reward = te.R[np.arange(te.N), tau]
                arrays[f"tau_{name}"] = tau.astype(np.int16)
                arrays[f"reward_{name}"] = reward
                arrays[f"exit_date_{name}"] = fold.test.metadata["decision_dates"][np.arange(te.N), tau]
                # Keep economic returns separate from a possibly nonlinear training reward.
                arrays[f"net_return_{name}"] = fold.test.metadata["net_returns"][np.arange(te.N), tau]
                arrays[f"objective_{name}"] = reward
                row = dict(fold=fold.number, method=name, seed=seed, n_train=tr.N,
                    n_validation=va.N, n_test=te.N, mean_reward=float(reward.mean()),
                    mean_stop_index=float(tau.mean()), smoke=smoke, **fold.boundaries)
                rows.append(row)
                print(json.dumps(row), flush=True)
            np.savez_compressed(output_dir / f"fold_{fold.number}_seed_{seed}.npz", **arrays)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "folds.json").write_text(json.dumps([
        dict(number=f.number, n_train=f.train.problem.N, n_validation=f.validation.problem.N,
             n_test=f.test.problem.N, **f.boundaries) for f in folds], indent=2))
    summary = ["# P4 walk-forward harness results", "",
        "SMOKE VALIDATION ONLY: deliberately shortened neural training." if smoke else
        "Out-of-sample mean per-window reward. Overlapping trades are not portfolio returns.", "",
        "| Fold | Method | Seed | Test windows | Mean reward |", "| --- | --- | --- | --- | --- |"]
    summary += [f"| {r['fold']} | {r['method']} | {r['seed']} | {r['n_test']} | {r['mean_reward']:.6f} |" for r in rows]
    summary += ["", "No iid significance claims: use date-block resampling in P5.",
        "Learned methods retain their existing global date-zero decision; heterogeneous entry states are a limitation."]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n")
    return frame


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", choices=["nifty50", "sp100", "crypto", "csv"], default="sp100")
    p.add_argument("--symbols", nargs="+", help="Explicit Yahoo or Binance symbols; crypto accepts top10")
    p.add_argument("--csv", type=Path)
    p.add_argument("--context-csv", type=Path, help="Date-indexed local index/VIX levels, lagged before alignment")
    p.add_argument("--context", choices=["local", "none"], default="local",
                   help="Fetch lagged local index/VIX for equities; crypto has no cross-market context")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2025-01-01", help="Exclusive; freeze dates for reproducibility")
    p.add_argument("--calendar-days", type=int, default=30)
    p.add_argument("--sessions", type=int, help="Optional fixed-session horizon instead of calendar days")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--venue", choices=["nse", "us", "crypto", "perpetual", "zero"])
    p.add_argument("--cost-multiplier", type=float, default=1.0)
    p.add_argument("--cost-json", type=Path, help="JSON keyword arguments for CostModel")
    p.add_argument("--reward", choices=["linear", "exponential"], default="linear")
    p.add_argument("--risk-aversion", type=float, default=5.0)
    p.add_argument("--train-months", type=int, default=36)
    p.add_argument("--validation-months", type=int, default=6)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--embargo-days", type=int, default=0)
    p.add_argument("--max-folds", type=int)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--methods", nargs="+", default=["Hold", "TakeProfitStopLoss", "TrailingStop", "LSM"],
        choices=["Hold", "TakeProfitStopLoss", "TrailingStop", "LSM", "DOS", "FQI-MLP", "FQI-GRU", "OSPG-MLP", "OSPG-GRU"])
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Shortened neural training: validation, not research results")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=CACHE)
    p.add_argument("--output", type=Path, help="Default results/p4/<market>_<configuration hash>")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("max-folds must be positive")
    venue = args.venue or {"nifty50": "nse", "sp100": "us", "crypto": "crypto", "csv": "zero"}[args.market]
    costs = (CostModel(**json.loads(args.cost_json.read_text())).scaled(args.cost_multiplier)
             if args.cost_json else venue_costs(venue, args.cost_multiplier))
    if args.market == "csv":
        if not args.csv:
            raise ValueError("--csv is required for --market csv")
        prices = load_prices_csv(args.csv)
        prices = prices.loc[(prices.index >= args.start) & (prices.index < args.end)]
    elif args.market == "crypto":
        symbols = "top10" if args.symbols == ["top10"] else (args.symbols or ("BTC/USDT", "ETH/USDT"))
        prices = load_crypto(symbols, args.start, args.end, cache_dir=args.cache_dir,
                             refresh=args.refresh, offline=args.offline)
    else:
        prices = load_equities(args.market, args.start, args.end, symbols=args.symbols,
            cache_dir=args.cache_dir, refresh=args.refresh, offline=args.offline)
    context = load_prices_csv(args.context_csv) if args.context_csv else None
    if context is None and args.context == "local" and args.market in {"nifty50", "sp100"}:
        context_symbols = ["^NSEI", "^INDIAVIX"] if args.market == "nifty50" else ["^GSPC", "^VIX"]
        context = load_equities(symbols=context_symbols, start=args.start, end=args.end,
            cache_dir=args.cache_dir, refresh=args.refresh, offline=args.offline)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"output", "refresh", "offline", "cache_dir"}}
    # Include actual source bytes: refreshes cannot quietly reuse a previous output path.
    config["price_sha256"] = sha256(prices.to_csv().encode()).hexdigest()
    config["costs"] = asdict(costs)
    if context is not None:
        config["context_sha256"] = sha256(context.to_csv().encode()).hexdigest()
    digest = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output or ROOT / "results" / "p4" / f"{args.market}_{digest}"
    windows, manifest = prepare(prices, market=args.market, output_dir=output,
        calendar_days=args.calendar_days, sessions=args.sessions, lookback=args.lookback,
        stride=args.stride, costs=costs, context=context, reward=args.reward, risk_aversion=args.risk_aversion,
        calendar="crypto" if venue in {"crypto", "perpetual"} else "equity")
    if context is not None:
        context.to_csv(output / "context.csv")
        (output / "context_manifest.json").write_text(json.dumps(context.attrs.get("manifest", {}), indent=2))
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"Prepared {windows.problem.N} windows, {len(manifest['assets'])} assets: {output}", flush=True)
    if not args.prepare_only:
        import torch
        torch.set_num_threads(3)
        folds = walk_forward_folds(windows, train_months=args.train_months,
            validation_months=args.validation_months, test_months=args.test_months,
            embargo_days=args.embargo_days)
        if args.max_folds:
            folds = folds[:args.max_folds]
        evaluate_folds(windows, folds, args.methods, output_dir=output, seeds=args.seeds, smoke=args.smoke)
    return output


if __name__ == "__main__":
    main()
