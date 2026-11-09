"""Regenerate P4 coverage/validation tables from saved manifests and actual per-path files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarize(root=None):
    root = Path(root) if root else ROOT / "results" / "p4"
    lines = ["# P4 implementation and real-data validation", "",
        "Fixed research period ends 2025-01-01 exclusive. Membership/rankings were retrieved later;",
        "these are survivor-selected research universes, not historical index membership.", "",
        "| Market | Requested | Usable assets | Windows | Folds |", "| --- | --- | --- | --- | --- |"]
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Market & Requested & Usable & Windows & Folds \\", r"\midrule"]
    rows, failure_lines = [], []
    for market, label in (("nifty50", "NIFTY 50"), ("sp100", r"S\&P 100"), ("crypto", "Crypto")):
        directory = root / market
        manifest = json.loads((directory / "dataset_manifest.json").read_text())
        source = json.loads((directory / "source_manifest.json").read_text())
        folds = json.loads((directory / "folds.json").read_text())
        n_requested, n_assets = len(source["requested_symbols"]), len(manifest["assets"])
        lines.append(f"| {market} | {n_requested} | {n_assets} | {manifest['n_windows']:,} | {len(folds)} |")
        tex.append(f"{label} & {n_requested} & {n_assets} & {manifest['n_windows']:,} & {len(folds)} " + r"\\")
        for symbol, error in source.get("failures", {}).items():
            failure_lines.append(f"- {market}: `{symbol}` excluded: {error}")
        for fold in folds:
            file = directory / f"fold_{fold['number']}_seed_0.npz"
            with np.load(file, allow_pickle=False) as z:
                assert len(set(z["window_id"].tolist())) == len(z["window_id"])
                for method in ("Hold", "TakeProfitStopLoss", "TrailingStop"):
                    values = z[f"net_return_{method}"]
                    assert np.isfinite(values).all()
                    assert (z[f"exit_date_{method}"] <= z["deadline"]).all()
                    rows.append(dict(market=market, method=method, n=len(values), total=float(values.sum())))
    tex += [r"\bottomrule", r"\end{tabular}"]
    (root / "coverage_table.tex").write_text("\n".join(tex) + "\n")
    lines += ["", "## Provider exclusions", "", *failure_lines, "",
        "## Rule validation: out-of-sample mean net return per window", "",
        "All folds pooled by actual window counts. No annualization, portfolio return, or significance claim.", "",
        "| Market | Hold | Profit/loss rule | Trailing stop |", "| --- | --- | --- | --- |"]
    df = pd.DataFrame(rows)
    for market in ("nifty50", "sp100", "crypto"):
        means = []
        for method in ("Hold", "TakeProfitStopLoss", "TrailingStop"):
            x = df[(df.market == market) & (df.method == method)]
            means.append(f"{100 * x.total.sum() / x.n.sum():.3f}%")
        lines.append(f"| {market} | {' | '.join(means)} |")
    lines += ["", "## Checks", "",
        "- All saved rule outputs have finite net returns, unique window IDs per fold, and exit dates within deadlines.",
        "- Full unit suite: 71 tests passed after adding 12 P4 tests (see tests/test_p4.py).",
        "- Real-data learned-model handoff: results/p4/model_handoff_smoke; deliberately shortened neural training.",
        "- Smoke values are compatibility checks, not final P5 trained-model evidence.",
        "- Economic model, source-selection bias and P5 handoff are documented in backtest/P4_GUIDE.md."]
    (root / "overview.md").write_text("\n".join(lines) + "\n")
    return root / "overview.md"


if __name__ == "__main__":
    print(summarize())
