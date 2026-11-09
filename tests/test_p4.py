"""P4 economic, temporal and end-to-end invariants; entirely offline fixtures."""
import json
import numpy as np
import pandas as pd
import pytest
from backtest.costs import CostModel, venue_costs
from backtest.rules import HoldToEnd, TakeProfitStopLoss, TrailingStop
from backtest.run import prepare, evaluate_folds
from common.problem import check_causal
from data.clean import clean_prices
from data.features import build_features
from data.loaders import _load, load_crypto
from data.splits import split_at, walk_forward_folds
from data.windows import MarketWindows, make_windows


def prices(n=1200, daily=False):
    rng = np.random.default_rng(8)
    dates = pd.date_range("2018-01-01", periods=n, freq="D" if daily else "B")
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (n, 2)), axis=0)),
                        index=dates, columns=["A", "B"])


def test_costs_economic_accounting():
    cost = CostModel(buy_bps=10, sell_bps=20, funding_bps_per_day=1)
    assert cost.net_return(1.1, 10) == pytest.approx(1.1 * .998 / 1.001 - 1 - .001)
    assert cost.net_return(1, 0) < 0
    assert venue_costs("us", 0).net_return(1.1, 5) == pytest.approx(.1)
    with pytest.raises(ValueError):
        CostModel(sell_bps=10000)


def test_cleaning_no_backward_fill_no_deleting_crashes():
    p = prices(100)
    p.iloc[0:3, 0] = np.nan
    p.iloc[40, 0] = -1
    p.iloc[70:, 1] *= .4
    p["USDT"] = 1.
    q, log = clean_prices(p)
    assert "USDT" not in q
    assert q.iloc[:3, 0].isna().all() and np.isnan(q.iloc[40, 0])
    assert q.iloc[70, 1] == p.iloc[70, 1]
    assert any(row["action"] == "invalid_price_to_missing" for row in log)


def test_crypto_calendar_preserves_missing_days():
    p = prices(100, daily=True).drop(pd.Timestamp("2018-02-03"))
    q, _ = clean_prices(p, calendar="crypto")
    assert q.loc["2018-02-03"].isna().all()


def test_features_and_cleaning_prefix_invariant():
    p = prices(250)
    changed = p.copy()
    changed.iloc[170:] *= 3
    q1, _ = clean_prices(p)
    q2, _ = clean_prices(changed)
    f1, f2 = build_features(q1), build_features(q2)
    np.testing.assert_allclose(f1["A"].iloc[:170], f2["A"].iloc[:170], equal_nan=True)
    context = pd.DataFrame({"index": [10., 20., 30.]}, index=p.index[[0, 2, 4]])
    f = build_features(p, context=context)["A"]
    assert np.isnan(f.loc[p.index[0], "context_index"])
    assert f.loc[p.index[3], "context_index"] == 10


def test_calendar_deadline_padding_and_roundtrip(tmp_path):
    ds = make_windows(prices(180), calendar_days=30, costs=CostModel(10, 10))
    p, m = ds.problem, ds.metadata
    assert p.X.shape[:2] == p.R.shape and p.H == 30
    assert (m["end_date"] <= m["deadline"]).all()
    assert ((m["deadline"] - m["entry_date"]) == np.timedelta64(30, "D")).all()
    i, end = 0, m["terminal_index"][0]
    assert p.R[i, 0] < 0
    np.testing.assert_allclose(p.R[i, end:], p.R[i, end])
    assert (m["decision_dates"][i, end:] == m["end_date"][i]).all()
    ds.save(tmp_path / "windows.npz")
    loaded = MarketWindows.load(tmp_path / "windows.npz")
    np.testing.assert_array_equal(loaded.metadata["window_id"], m["window_id"])
    np.testing.assert_allclose(loaded.problem.X, p.X)


def test_windows_no_future_features_even_when_future_rewards_change():
    p = prices(160)
    changed = p.copy()
    changed.iloc[100:] *= 2
    a, b = make_windows(p), make_windows(changed)
    for window_id in a.metadata["window_id"]:
        i = np.flatnonzero(a.metadata["window_id"] == window_id)[0]
        j = np.flatnonzero(b.metadata["window_id"] == window_id)[0]
        before = a.metadata["decision_dates"][i] < p.index[100].to_datetime64()
        np.testing.assert_allclose(a.problem.X[i, before], b.problem.X[j, before])
        np.testing.assert_allclose(a.problem.R[i, before], b.problem.R[j, before])


def test_missing_window_is_not_bridged():
    p = prices(150)
    missing_date = p.index[75]
    p.loc[missing_date, "A"] = np.nan
    ds = make_windows(p)
    m = ds.metadata
    selected = m["asset_id"] == "A"
    assert not ((m["entry_date"][selected] <= missing_date.to_datetime64()) &
                (m["end_date"][selected] >= missing_date.to_datetime64())).any()


def test_purged_folds_and_aligned_metadata():
    ds = make_windows(prices(), stride=5)
    folds = walk_forward_folds(ds, train_months=18, validation_months=6, test_months=6)
    seen = set()
    for fold in folds:
        tr, va, te = [part.metadata for part in (fold.train, fold.validation, fold.test)]
        assert tr["deadline"].max() < va["entry_date"].min()
        assert va["deadline"].max() < te["entry_date"].min()
        assert not (seen & set(te["window_id"]))
        seen.update(te["window_id"])
        assert len(te["asset_id"]) == fold.test.problem.N


def test_rule_known_path_and_causality():
    x = np.zeros((1, 5, 3))
    x[0, :, 0] = [0, .02, .06, .01, .03]
    x[0, :, 1] = [0, 0, 0, -.06, -.04]
    r = x[:, :, 0].copy()
    assert HoldToEnd().stop(x, r)[0] == 4
    assert TakeProfitStopLoss().stop(x, r)[0] == 2
    assert TrailingStop().stop(x, r)[0] == 3
    ds = make_windows(prices(200))
    for rule in (HoldToEnd(), TakeProfitStopLoss(), TrailingStop()):
        assert check_causal(rule, ds.problem)


def test_loader_cache_offline_and_failures(tmp_path):
    dates = pd.date_range("2020-01-01", periods=3)
    calls = []
    def fetch(symbol, start, end):
        calls.append(symbol)
        if symbol == "bad":
            raise ValueError("fixture missing asset")
        return pd.Series([1., 2., 3.], index=dates)
    frame = _load(["good", "bad"], "2020-01-01", "2020-01-04", "fixture", fetch, tmp_path, False, False)
    assert "bad" in frame.attrs["manifest"]["failures"]
    old = len(calls)
    cached = _load(["good", "bad"], "2020-01-01", "2020-01-04", "fixture", fetch, tmp_path, False, True)
    assert len(calls) == old and cached.shape == (3, 1)
    with pytest.raises(FileNotFoundError):
        _load(["new"], "2020-01-01", "2020-01-04", "fixture", fetch, tmp_path, False, True)


def test_binance_pagination_and_end_exclusive(monkeypatch, tmp_path):
    from data import loaders
    calls = []
    start = int(pd.Timestamp("2020-01-01").timestamp() * 1000)
    class Response:
        def __init__(self, data): self.data = data
        def json(self): return self.data
    def get(url, params):
        calls.append(params)
        if len(calls) == 1:
            return Response([[start, 0, 0, 0, "100"], [start + 86400000, 0, 0, 0, "110"]])
        return Response([])
    monkeypatch.setattr(loaders, "_get", get)
    frame = load_crypto(["BTC/USDT"], "2020-01-01", "2020-01-05", cache_dir=tmp_path)
    assert frame.shape == (2, 1)
    assert calls[1]["startTime"] == start + 2 * 86400000


def test_end_to_end_dated_results_and_utility(tmp_path):
    ds, manifest = prepare(prices(), market="fixture", output_dir=tmp_path,
                           costs=venue_costs("us"), reward="exponential")
    folds = walk_forward_folds(ds, train_months=18, validation_months=6, test_months=6)[:1]
    result = evaluate_folds(ds, folds, ["Hold", "TakeProfitStopLoss", "TrailingStop"], output_dir=tmp_path)
    assert len(result) == 3
    with np.load(tmp_path / "fold_0_seed_0.npz", allow_pickle=False) as z:
        assert (z["exit_date_Hold"] <= z["deadline"]).all()
        np.testing.assert_array_equal(z["tau_Hold"], z["terminal_index"])
        assert "net_return_Hold" in z
        assert not np.allclose(z["net_return_Hold"], z["reward_Hold"])
