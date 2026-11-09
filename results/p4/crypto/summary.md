# P4 walk-forward harness results

Out-of-sample mean per-window reward. Overlapping trades are not portfolio returns.

| Fold | Method | Seed | Test windows | Mean reward |
| --- | --- | --- | --- | --- |
| 0 | Hold | 0 | 248 | 0.207992 |
| 0 | TakeProfitStopLoss | 0 | 248 | 0.008744 |
| 0 | TrailingStop | 0 | 248 | 0.050293 |
| 1 | Hold | 0 | 240 | -0.024599 |
| 1 | TakeProfitStopLoss | 0 | 240 | -0.004282 |
| 1 | TrailingStop | 0 | 240 | -0.001871 |
| 2 | Hold | 0 | 248 | -0.079623 |
| 2 | TakeProfitStopLoss | 0 | 248 | -0.004222 |
| 2 | TrailingStop | 0 | 248 | -0.012303 |
| 3 | Hold | 0 | 240 | 0.016259 |
| 3 | TakeProfitStopLoss | 0 | 240 | 0.009092 |
| 3 | TrailingStop | 0 | 240 | 0.013276 |
| 4 | Hold | 0 | 248 | 0.003755 |
| 4 | TakeProfitStopLoss | 0 | 248 | -0.004104 |
| 4 | TrailingStop | 0 | 248 | -0.000630 |
| 5 | Hold | 0 | 237 | 0.122534 |
| 5 | TakeProfitStopLoss | 0 | 237 | 0.021612 |
| 5 | TrailingStop | 0 | 237 | 0.047942 |
| 6 | Hold | 0 | 217 | 0.000512 |
| 6 | TakeProfitStopLoss | 0 | 217 | -0.013012 |
| 6 | TrailingStop | 0 | 217 | -0.007503 |

No iid significance claims: use date-block resampling in P5.
Learned methods retain their existing global date-zero decision; heterogeneous entry states are a limitation.
