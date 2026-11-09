# P4 walk-forward harness results

SMOKE VALIDATION ONLY: deliberately shortened neural training.

| Fold | Method | Seed | Test windows | Mean reward |
| --- | --- | --- | --- | --- |
| 0 | Hold | 0 | 44 | -0.029552 |
| 0 | LSM | 0 | 44 | -0.008900 |
| 0 | DOS | 0 | 44 | -0.029049 |
| 0 | FQI-MLP | 0 | 44 | -0.003471 |
| 0 | FQI-GRU | 0 | 44 | -0.011739 |
| 0 | OSPG-MLP | 0 | 44 | -0.029552 |
| 0 | OSPG-GRU | 0 | 44 | -0.029552 |

No iid significance claims: use date-block resampling in P5.
Learned methods retain their existing global date-zero decision; heterogeneous entry states are a limitation.
