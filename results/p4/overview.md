# P4 implementation and real-data validation

Fixed research period ends 2025-01-01 exclusive. Membership/rankings were retrieved later;
these are survivor-selected research universes, not historical index membership.

| Market | Requested | Usable assets | Windows | Folds |
| --- | --- | --- | --- | --- |
| nifty50 | 50 | 50 | 20,014 | 12 |
| sp100 | 101 | 100 | 48,444 | 12 |
| crypto | 10 | 8 | 3,590 | 7 |

## Provider exclusions

- sp100: `SNDK` excluded: $SNDK: Data doesn't exist for startDate = 1420088400, endDate = 1735707600
- crypto: `FIGR_HELOC/USDT` excluded: 400 Client Error: Bad Request for url: https://data-api.binance.vision/api/v3/klines?symbol=FIGR_HELOCUSDT&interval=1d&startTime=1483228800000&endTime=1735689599999&limit=1000
- crypto: `HYPE/USDT` excluded: No observations in requested interval

## Rule validation: out-of-sample mean net return per window

All folds pooled by actual window counts. No annualization, portfolio return, or significance claim.

| Market | Hold | Profit/loss rule | Trailing stop |
| --- | --- | --- | --- |
| nifty50 | 1.868% | 0.575% | 1.118% |
| sp100 | 1.305% | 0.450% | 0.835% |
| crypto | 3.571% | 0.212% | 1.295% |

## Checks

- All saved rule outputs have finite net returns, unique window IDs per fold, and exit dates within deadlines.
- Full unit suite: 71 tests passed after adding 12 P4 tests (see tests/test_p4.py).
- Real-data learned-model handoff: results/p4/model_handoff_smoke; deliberately shortened neural training.
- Smoke values are compatibility checks, not final P5 trained-model evidence.
- Economic model, source-selection bias and P5 handoff are documented in backtest/P4_GUIDE.md.
