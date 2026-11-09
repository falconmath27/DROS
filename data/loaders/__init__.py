"""Cached adjusted daily prices. Universe snapshots are not historical membership."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import time
import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "cache"
NIFTY_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
SP100_URL = "https://en.wikipedia.org/wiki/S%26P_100"
STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "FDUSD", "USDD",
               "PYUSD", "USDE", "USD1", "USDS", "USDG", "RLUSD", "FRAX", "LUSD"}


def _get(url, **kwargs):
    r = requests.get(url, timeout=45, headers={"User-Agent": "DROS-academic-project/1.0"}, **kwargs)
    r.raise_for_status()
    return r


def universe_symbols(universe, cache_dir=CACHE, refresh=False):
    """Fetch/freeze retrieval-date membership. Supply symbols= for a dated custom list."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"universe_{universe}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())["symbols"]
    if universe == "nifty50":
        frame = pd.read_csv(StringIO(_get(NIFTY_URL).text))
        symbols = [str(s) + ".NS" for s in frame["Symbol"]]
        source = NIFTY_URL
    elif universe == "sp100":
        tables = pd.read_html(StringIO(_get(SP100_URL).text))
        frame = next(t for t in tables if "Symbol" in t.columns and "Name" in t.columns)
        symbols = [str(s).replace(".", "-") for s in frame["Symbol"]]
        source = SP100_URL
    else:
        raise ValueError("universe must be nifty50 or sp100; use symbols= for custom lists")
    path.write_text(json.dumps(dict(symbols=symbols, source=source,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        limitation="Current membership; historical selection and survivorship bias."), indent=2))
    return symbols


def top_crypto_symbols(cache_dir=CACHE, refresh=False):
    """Current market-cap top ten non-stable, unwrapped assets. Selection is not historical."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "universe_crypto_top10.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())["symbols"]
    url = "https://api.coingecko.com/api/v3/coins/markets"
    coins = _get(url, params=dict(vs_currency="usd", order="market_cap_desc", per_page=100,
                                 page=1, sparkline="false")).json()
    chosen = []
    for coin in coins:
        symbol, name = coin["symbol"].upper(), coin["name"].lower()
        if symbol in STABLECOINS or "USD" in symbol or any(w in name for w in
                ("wrapped", "staked", "bridged", "stablecoin")):
            continue
        if symbol + "/USDT" not in chosen:
            chosen.append(symbol + "/USDT")
        if len(chosen) == 10:
            break
    if len(chosen) != 10:
        raise ValueError("Could not identify ten eligible assets")
    path.write_text(json.dumps(dict(symbols=chosen, source=url,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        limitation="Retrieval-date market-cap selection; not historical membership."), indent=2))
    return chosen


def _normalize(series):
    series = series.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(series.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    series.index = idx.normalize()
    return series[~series.index.duplicated(keep="last")].sort_index()


def _load(symbols, start, end, source, fetch, cache_dir, refresh, offline):
    if isinstance(symbols, str):
        raise ValueError("Pass symbols as a sequence, not a single string")
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize() if end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if end <= start:
        raise ValueError("end must be after start; end is exclusive")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise ValueError("No symbols requested")
    key = sha256(json.dumps([source, symbols, str(start), str(end)]).encode()).hexdigest()[:16]
    path = cache_dir / f"{source}_{key}.csv"
    manifest_path = path.with_suffix(".json")
    if path.exists() and not refresh:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame.attrs["manifest"] = json.loads(manifest_path.read_text())
        return frame
    if offline:
        raise FileNotFoundError(f"No cached request at {path}; run online with identical dates/symbols")
    def one(symbol):
        error = ""
        for attempt in range(3):
            try:
                series = _normalize(fetch(symbol, start, end))
                series = series.loc[(series.index >= start) & (series.index < end)].dropna()
                if series.empty:
                    raise ValueError("No observations in requested interval")
                return symbol, series.rename(symbol), None
            except Exception as exc:
                error = str(exc)
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        return symbol, None, error
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, symbols))
    failures = {s: err for s, _, err in records if err}
    series = [x for _, x, err in records if not err]
    manifest = dict(source=source, requested_symbols=symbols, failures=failures,
        start=str(start.date()), end_exclusive=str(end.date()),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        adjustment="split/dividend adjusted close" if source == "yahoo" else "spot daily close; UTC sessions",
        universe_warning="Fixed supplied/retrieval-date universe; not historical membership.")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if not series:
        raise RuntimeError(f"All downloads failed; details: {manifest_path}")
    frame = pd.concat(series, axis=1).sort_index()
    frame.index.name = "date"
    frame.to_csv(path)
    manifest["cache_file"] = str(path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    frame.attrs["manifest"] = manifest
    return frame


def load_equities(universe="nifty50", start="2005-01-01", end=None, *, symbols=None,
                  cache_dir=CACHE, refresh=False, offline=False):
    """Yahoo adjusted daily closes; explicit symbols also support indices, FX and gold."""
    import yfinance as yf
    if offline and refresh:
        raise ValueError("offline and refresh cannot be combined")
    if offline and symbols is None and not (Path(cache_dir) / f"universe_{universe}.json").exists():
        raise FileNotFoundError("Offline universe snapshot is missing; supply explicit symbols or fetch online")
    symbols = symbols if symbols is not None else universe_symbols(universe, cache_dir, refresh)
    def fetch(symbol, start, end):
        return yf.Ticker(symbol).history(start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True,
            actions=True, raise_errors=True)["Close"]
    return _load(symbols, start, end, "yahoo", fetch, cache_dir, refresh, offline)


def load_crypto(symbols=("BTC/USDT", "ETH/USDT"), start="2017-01-01", end=None, *,
                cache_dir=CACHE, refresh=False, offline=False, endpoint="https://data-api.binance.vision"):
    """Paginated public Binance spot daily klines; no account/private endpoints."""
    if offline and refresh:
        raise ValueError("offline and refresh cannot be combined")
    if isinstance(symbols, str):
        if symbols != "top10":
            raise ValueError("Use a list of symbols or 'top10'")
        if offline and not (Path(cache_dir) / "universe_crypto_top10.json").exists():
            raise FileNotFoundError("Offline crypto universe snapshot is missing")
        symbols = top_crypto_symbols(cache_dir, refresh)
    def fetch(symbol, start, end):
        pair = symbol.replace("/", "").upper()
        if pair.removesuffix("USDT") in STABLECOINS:
            raise ValueError("Stablecoins excluded")
        cursor, last = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        rows = []
        while cursor < last:
            batch = _get(endpoint + "/api/v3/klines", params=dict(symbol=pair, interval="1d",
                startTime=cursor, endTime=last - 1, limit=1000)).json()
            if not isinstance(batch, list):
                raise ValueError(f"Invalid kline response: {batch}")
            if not batch:
                break
            if int(batch[-1][0]) < cursor:
                raise ValueError("Kline pagination did not advance")
            rows.extend(batch)
            cursor = int(batch[-1][0]) + 86_400_000
        return pd.Series([float(row[4]) for row in rows],
            index=pd.to_datetime([row[0] for row in rows], unit="ms", utc=True).tz_localize(None))
    return _load(symbols, start, end, "binance", fetch, cache_dir, refresh, offline)


def load_prices_csv(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.attrs["manifest"] = dict(source=str(Path(path).resolve()), adjustment="user supplied; verify adjustments")
    return frame
