"""Focused validation for the P5 paired date-block statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.bootstrap import (
    _date_block_groups,
    average_differences_across_seeds,
    align_method_scores,
    analyze_comparison_families,
    analyze_contrast,
    holm_adjust,
    load_fold_results,
)


ROOT = Path(__file__).resolve().parents[1]


def toy_results(shift=0.0, seeds=(0, 1, 2)):
    rows = []
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    for fold, selected_dates in enumerate((dates[:6], dates[6:])):
        for seed in seeds:
            for date in selected_dates:
                for asset, baseline in (("A", 0.01), ("B", -0.005)):
                    window_id = f"toy:{asset}:{date.date()}"
                    right = baseline + seed * 0.001
                    for method, score in (("OSPG-GRU", right + shift),
                                          ("OSPG-MLP", right)):
                        rows.append({
                            "market": "toy",
                            "fold": fold,
                            "seed": seed,
                            "window_id": window_id,
                            "entry_date": date,
                            "asset_id": asset,
                            "method": method,
                            "score": score,
                        })
    return pd.DataFrame(rows)


def test_zero_difference_and_seed_reproducibility():
    results = toy_results()
    first = analyze_contrast(
        results, "OSPG-GRU", "OSPG-MLP", block_days=(2, 3, 5),
        primary_block_days=3, n_boot=300, seed=17, expected_seeds=[0, 1, 2])
    second = analyze_contrast(
        results, "OSPG-GRU", "OSPG-MLP", block_days=(2, 3, 5),
        primary_block_days=3, n_boot=300, seed=17, expected_seeds=[0, 1, 2])
    assert first["estimate"] == pytest.approx(0)
    assert first["ci_lower"] == pytest.approx(0)
    assert first["ci_upper"] == pytest.approx(0)
    assert first["p_value"] == pytest.approx(1)
    assert first["n_windows"] == 24
    assert first["n_dates"] == 12
    assert first["block_length_sensitivity"] == second["block_length_sensitivity"]


def test_known_shift_multiple_assets_and_per_seed_summary():
    result = analyze_contrast(
        toy_results(shift=0.02), "OSPG-GRU", "OSPG-MLP",
        block_days=(2, 3, 5), primary_block_days=3, n_boot=300, seed=5,
        expected_seeds=[0, 1, 2])
    assert result["estimate"] == pytest.approx(0.02)
    assert result["ci_lower"] == pytest.approx(0.02)
    assert result["ci_upper"] == pytest.approx(0.02)
    assert result["p_value"] == pytest.approx(1 / 301)
    assert len(result["per_seed"]) == 3
    assert all(row["estimate"] == pytest.approx(0.02) for row in result["per_seed"])
    assert all(row["n_windows"] == 24 for row in result["per_seed"])


def test_missing_duplicate_and_nonfinite_matches_rejected():
    results = toy_results()
    missing_index = results.index[results["method"] == "OSPG-MLP"][0]
    missing = results.drop(missing_index)
    with pytest.raises(ValueError, match="Missing paired"):
        align_method_scores(missing, "OSPG-GRU", "OSPG-MLP")

    duplicate = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        align_method_scores(duplicate, "OSPG-GRU", "OSPG-MLP")

    nonfinite = results.copy()
    nonfinite.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        align_method_scores(nonfinite, "OSPG-GRU", "OSPG-MLP")


def test_missing_seed_for_a_window_rejected():
    results = toy_results()
    key = results.iloc[0]["window_id"]
    results = results.loc[~((results["window_id"] == key) & (results["seed"] == 2))]
    with pytest.raises(ValueError, match="every expected seed"):
        analyze_contrast(
            results, "OSPG-GRU", "OSPG-MLP", block_days=(3,),
            primary_block_days=3, n_boot=20, seed=1, expected_seeds=[0, 1, 2])


def test_holm_and_configured_comparison_families():
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    results = toy_results(shift=0.01)
    config = {
        "seeds": [0, 1, 2],
        "contrasts": {
            "primary": {"left": "OSPG-GRU", "right": "OSPG-MLP"},
            "secondary": [{"left": "OSPG-GRU", "right": "OSPG-MLP"}],
        },
    }
    families = analyze_comparison_families(
        results, config, block_days=(3,), primary_block_days=3,
        n_boot=50, seed=2)
    assert families["primary"][0]["comparison_family"] == "primary"
    assert families["primary"][0]["p_value_holm"] == families["primary"][0]["p_value"]
    assert families["secondary"][0]["comparison_family"] == "secondary"


def test_terminal_date_is_available_to_boundary_block():
    paired = align_method_scores(toy_results(shift=0.01), "OSPG-GRU", "OSPG-MLP")
    averaged = average_differences_across_seeds(paired, [0, 1, 2])
    groups = _date_block_groups(averaged, block_days=3)
    for date_sums, _, blocks in groups:
        assert any(len(date_sums) - 1 in block for block in blocks)


def test_load_one_saved_p4_rule_fold():
    path = ROOT / "results" / "p4" / "nifty50" / "fold_0_seed_0.npz"
    results = load_fold_results(
        path, methods=["Hold", "TrailingStop"], metric="net_return")
    paired = align_method_scores(results, "Hold", "TrailingStop")
    assert len(paired) == 799
    assert paired["window_id"].is_unique
    assert paired["market"].eq("nifty50").all()
    assert paired["fold"].eq(0).all() and paired["seed"].eq(0).all()
    assert np.isfinite(paired["diff"]).all()


def test_full_grid_coverage_requires_valid_artifact(tmp_path):
    from experiments.backtest.full_grid import collect_coverage

    scenario = tmp_path / "sp100" / "linear" / "cost_1x"
    scenario.mkdir(parents=True)
    metadata = {
        "window_id": np.array(["sp100:A:2024-01-01"]),
        "terminal_index": np.array([2]),
        "end_date": np.array(["2024-01-03"], dtype="datetime64[D]"),
    }
    np.savez_compressed(
        scenario / "fold_0_seed_0.npz",
        **metadata,
        tau_Hold=np.array([2]),
        reward_Hold=np.array([0.01]),
        objective_Hold=np.array([0.01]),
        net_return_Hold=np.array([0.01]),
        exit_date_Hold=np.array(["2024-01-03"], dtype="datetime64[D]"),
    )
    status = {
        "fold": 0,
        "seed": 0,
        "n_test": 1,
        "window_id_sha256": "window",
        "entry_date_sha256": "entry",
        "asset_id_sha256": "asset",
        "test_features_sha256": "features",
        "methods": {"Hold": {"status": "completed", "error": None}},
    }
    (scenario / "fold_0_seed_0.status.json").write_text(
        json.dumps(status))
    scope = {
        "markets": ["sp100"],
        "objectives": ["linear"],
        "cost_multipliers": [1.0],
        "folds_by_market": {"sp100": [0]},
        "seeds": [0],
        "methods": ["Hold"],
        "smoke": True,
    }
    coverage = collect_coverage(tmp_path, scope)
    assert coverage["passed"]
    assert coverage["counts"] == {
        "expected": 1, "completed": 1, "failed": 0, "pending": 0}


def test_full_grid_cost_labels():
    from experiments.backtest.full_grid import cost_label

    assert cost_label(0) == "cost_0x"
    assert cost_label(1.0) == "cost_1x"
    assert cost_label(2) == "cost_2x"
    with pytest.raises(ValueError):
        cost_label(-1)


def test_full_grid_progress_snapshot_and_line(tmp_path):
    from experiments.backtest.full_grid import (
        _format_duration,
        _progress_line,
        _progress_snapshot,
    )

    status = {
        "methods": {
            "OSPG-GRU": {"status": "completed"},
            "OSPG-MLP": {"status": "running"},
        },
    }
    (tmp_path / "fold_0_seed_0.status.json").write_text(json.dumps(status))
    snapshot = _progress_snapshot(
        tmp_path, [(0, 0), (0, 1)], ["OSPG-GRU", "OSPG-MLP"])
    assert snapshot["counts"] == {
        "completed": 1, "failed": 0, "running": 1, "pending": 2}
    assert snapshot["active"] == ["fold 0/seed 0: OSPG-MLP"]
    line = _progress_line(
        "Heartbeat", completed_jobs=0, total_jobs=2, workers=2,
        elapsed=15, snapshot=snapshot)
    assert "[P5] Heartbeat" in line
    assert "methods 1 complete, 0 failed, 1 running" in line
    assert "elapsed 0m 15s" in line
    assert "ETA calculating" in line
    assert _format_duration(3665) == "1h 01m"
