# Paired comparisons on identical test paths

diff = (reference method) - (other method), averaged over seeds; se = paired standard error; z = diff / se. |z| > 2 is a difference unlikely to be noise. E5 rows assume independent windows, which overlapping windows are not: use P5's block bootstrap there.

## E1 - put_S0=36 (reference: OSPG-GRU)

| other method | seeds | diff | se | z |
| --- | --- | --- | --- | --- |
| DOS | 3 | +0.0006 | 0.0029 | +0.2 |
| FQI-MLP | 3 | -0.0020 | 0.0023 | -0.9 |
| LSM | 3 | +0.0075 | 0.0029 | +2.6 |
| OSPG-MLP | 3 | -0.0075 | 0.0025 | -3.0 |

## E2 - maxcall_d=20 (reference: OSPG-GRU)

| other method | seeds | diff | se | z |
| --- | --- | --- | --- | --- |
| DOS | 3 | +0.0298 | 0.0347 | +0.9 |
| FQI-GRU | 3 | -0.0081 | 0.0328 | -0.2 |
| FQI-MLP | 3 | +0.0343 | 0.0369 | +0.9 |
| LSM | 3 | +0.0306 | 0.0318 | +1.0 |
| OSPG-MLP | 3 | -0.0861 | 0.0294 | -2.9 |

## E2 - maxcall_d=50 (reference: OSPG-GRU)

| other method | seeds | diff | se | z |
| --- | --- | --- | --- | --- |
| DOS | 3 | +0.1469 | 0.0377 | +3.9 |
| FQI-GRU | 3 | +0.0534 | 0.0344 | +1.6 |
| FQI-MLP | 3 | +0.1961 | 0.0365 | +5.4 |
| LSM | 3 | +0.0411 | 0.0323 | +1.3 |
| OSPG-MLP | 3 | -0.0635 | 0.0286 | -2.2 |

## E3 - geocall_d=20 (reference: OSPG-GRU)

| other method | seeds | diff | se | z |
| --- | --- | --- | --- | --- |
| FQI-MLP | 3 | +0.0663 | 0.0223 | +3.0 |
| LSM | 3 | +0.0420 | 0.0162 | +2.6 |
| OSPG-MLP | 3 | -0.0487 | 0.0133 | -3.7 |

## E3 - geocall_d=7 (reference: OSPG-GRU)

| other method | seeds | diff | se | z |
| --- | --- | --- | --- | --- |
| FQI-MLP | 3 | +0.0920 | 0.0198 | +4.6 |
| LSM | 3 | +0.0673 | 0.0145 | +4.6 |
| OSPG-MLP | 3 | -0.0179 | 0.0116 | -1.5 |

## E4 - rbergomi_put (reference: OSPG-GRU)

| other method | seeds | diff | se | z |
| --- | --- | --- | --- | --- |
| DOS | 2 | +0.0810 | 0.0179 | +4.5 |
| FQI-GRU | 2 | +0.0468 | 0.0179 | +2.6 |
| LSM | 2 | +0.0352 | 0.0152 | +2.3 |
| LSM-lags3 | 2 | +0.0390 | 0.0152 | +2.6 |
| OSPG-MLP | 2 | -0.0036 | 0.0133 | -0.3 |

## E5 - synthetic_market (reference: OSPG-GRU)

| other method | seeds | diff | se | z |
| --- | --- | --- | --- | --- |
| DOS | 2 | +0.0012 | 0.0004 | +3.4 |
| Hold | 2 | -0.0000 | 0.0000 | -0.4 |
| LSM | 2 | +0.0002 | 0.0002 | +1.1 |
| OSPG-MLP | 2 | +0.0005 | 0.0002 | +2.4 |

