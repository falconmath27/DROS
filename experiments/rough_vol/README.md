# Rough-volatility experiments (P3)

- Put configs and seeds for the fBm table and the American put under rough Bergomi here.
- `simulators/rbergomi.py` currently holds a **stand-in** rough Bergomi simulator written by
  P1/P2 (exact Cholesky scheme). Replace or validate it (European prices from the literature,
  hybrid scheme), keeping `rbergomi_put_problem` returning a `Problem`.
- E4 in `experiments/run_all.py` runs LSM, LSM with lags, DOS, FQI-GRU, OSPG-MLP and OSPG-GRU on
  it. Point `E4_rough_bergomi` at your simulator, add the H sweep (0.07, 0.1, 0.3, 0.5).
- Stand-in finding to re-check (see `results/paired.md`): OSPG beats the backward-induction
  methods by 0.035-0.08 (z = 2.3-4.5), but OSPG-GRU = OSPG-MLP (z = 0.3).
