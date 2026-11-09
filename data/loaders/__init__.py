"""Price loaders (P4).

Owner: P4 (market data and backtest harness). First deliverable, end of day 2: a harness
that turns any price series into train/test windows with features and rewards.

To do (from the project plan):
- [ ] NIFTY 50 and S&P 100 via yfinance, adjusted for splits and dividends
- [ ] BTC, ETH and top-10 crypto (Binance public data or ccxt)
- [ ] FX and gold only if time allows
Return a DataFrame: dates x tickers of adjusted closes.
simulators/market.py has a CSV loader (load_prices_csv) and synthetic stand-in prices.
"""


def load_equities(universe="nifty50", start="2005-01-01", end=None):
    raise NotImplementedError("P4")


def load_crypto(symbols=("BTC/USDT", "ETH/USDT"), start="2017-01-01", end=None):
    raise NotImplementedError("P4")
