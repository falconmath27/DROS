# P4 walk-forward harness results

Out-of-sample mean per-window reward. Overlapping trades are not portfolio returns.

| Fold | Method | Seed | Test windows | Mean reward |
| --- | --- | --- | --- | --- |
| 0 | Hold | 0 | 2037 | -0.010708 |
| 0 | TakeProfitStopLoss | 0 | 2037 | -0.005272 |
| 0 | TrailingStop | 0 | 2037 | -0.007176 |
| 1 | Hold | 0 | 2041 | 0.017555 |
| 1 | TakeProfitStopLoss | 0 | 2041 | 0.009128 |
| 1 | TrailingStop | 0 | 2041 | 0.012743 |
| 2 | Hold | 0 | 2058 | 0.033592 |
| 2 | TakeProfitStopLoss | 0 | 2058 | 0.014681 |
| 2 | TrailingStop | 0 | 2058 | 0.024482 |
| 3 | Hold | 0 | 2058 | 0.001932 |
| 3 | TakeProfitStopLoss | 0 | 2058 | 0.000211 |
| 3 | TrailingStop | 0 | 2058 | -0.000825 |
| 4 | Hold | 0 | 2165 | 0.037986 |
| 4 | TakeProfitStopLoss | 0 | 2165 | 0.010961 |
| 4 | TrailingStop | 0 | 2165 | 0.020545 |
| 5 | Hold | 0 | 2079 | 0.027906 |
| 5 | TakeProfitStopLoss | 0 | 2079 | 0.013085 |
| 5 | TrailingStop | 0 | 2079 | 0.018014 |
| 6 | Hold | 0 | 2079 | 0.010950 |
| 6 | TakeProfitStopLoss | 0 | 2079 | 0.003100 |
| 6 | TrailingStop | 0 | 2079 | 0.008221 |
| 7 | Hold | 0 | 2079 | -0.020456 |
| 7 | TakeProfitStopLoss | 0 | 2079 | -0.009036 |
| 7 | TrailingStop | 0 | 2079 | -0.014005 |
| 8 | Hold | 0 | 2079 | 0.000735 |
| 8 | TakeProfitStopLoss | 0 | 2079 | 0.000647 |
| 8 | TrailingStop | 0 | 2079 | 0.002879 |
| 9 | Hold | 0 | 2079 | 0.018268 |
| 9 | TakeProfitStopLoss | 0 | 2079 | 0.003479 |
| 9 | TrailingStop | 0 | 2079 | 0.010646 |
| 10 | Hold | 0 | 2079 | 0.020570 |
| 10 | TakeProfitStopLoss | 0 | 2079 | 0.005679 |
| 10 | TrailingStop | 0 | 2079 | 0.013032 |
| 11 | Hold | 0 | 1989 | 0.017076 |
| 11 | TakeProfitStopLoss | 0 | 1989 | 0.007162 |
| 11 | TrailingStop | 0 | 1989 | 0.011105 |

No iid significance claims: use date-block resampling in P5.
Learned methods retain their existing global date-zero decision; heterogeneous entry states are a limitation.
