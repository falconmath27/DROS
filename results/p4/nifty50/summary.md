# P4 walk-forward harness results

Out-of-sample mean per-window reward. Overlapping trades are not portfolio returns.

| Fold | Method | Seed | Test windows | Mean reward |
| --- | --- | --- | --- | --- |
| 0 | Hold | 0 | 799 | -0.015915 |
| 0 | TakeProfitStopLoss | 0 | 799 | -0.008294 |
| 0 | TrailingStop | 0 | 799 | -0.012787 |
| 1 | Hold | 0 | 940 | 0.021009 |
| 1 | TakeProfitStopLoss | 0 | 940 | 0.005732 |
| 1 | TrailingStop | 0 | 940 | 0.010901 |
| 2 | Hold | 0 | 752 | 0.024277 |
| 2 | TakeProfitStopLoss | 0 | 752 | 0.004398 |
| 2 | TrailingStop | 0 | 752 | 0.007894 |
| 3 | Hold | 0 | 940 | -0.004442 |
| 3 | TakeProfitStopLoss | 0 | 940 | 0.002167 |
| 3 | TrailingStop | 0 | 940 | 0.003885 |
| 4 | Hold | 0 | 857 | 0.059199 |
| 4 | TakeProfitStopLoss | 0 | 857 | 0.013159 |
| 4 | TrailingStop | 0 | 857 | 0.037180 |
| 5 | Hold | 0 | 288 | 0.008864 |
| 5 | TakeProfitStopLoss | 0 | 288 | -0.004606 |
| 5 | TrailingStop | 0 | 288 | -0.002619 |
| 6 | Hold | 0 | 1026 | 0.020517 |
| 6 | TakeProfitStopLoss | 0 | 1026 | 0.004832 |
| 6 | TrailingStop | 0 | 1026 | 0.011643 |
| 7 | Hold | 0 | 980 | -0.004611 |
| 7 | TakeProfitStopLoss | 0 | 980 | -0.006863 |
| 7 | TrailingStop | 0 | 980 | -0.005192 |
| 8 | Hold | 0 | 833 | 0.005227 |
| 8 | TakeProfitStopLoss | 0 | 833 | 0.007413 |
| 8 | TrailingStop | 0 | 833 | 0.005597 |
| 9 | Hold | 0 | 1029 | 0.031263 |
| 9 | TakeProfitStopLoss | 0 | 1029 | 0.016182 |
| 9 | TrailingStop | 0 | 1029 | 0.026085 |
| 10 | Hold | 0 | 843 | 0.041201 |
| 10 | TakeProfitStopLoss | 0 | 843 | 0.017410 |
| 10 | TrailingStop | 0 | 843 | 0.030064 |
| 11 | Hold | 0 | 850 | 0.031919 |
| 11 | TakeProfitStopLoss | 0 | 850 | 0.009957 |
| 11 | TrailingStop | 0 | 850 | 0.010456 |

No iid significance claims: use date-block resampling in P5.
Learned methods retain their existing global date-zero decision; heterogeneous entry states are a limitation.
