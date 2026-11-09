# Backtest experiments (P5)

- Configs and seeds for the real-data runs: GRU-OSPG, MLP-OSPG, Longstaff-Schwartz and the
  rules, every asset class, walk-forward.
- Two rewards: linear return and risk-sensitive (exponential utility or mean-variance).
- E5 in `experiments/run_all.py` is the stand-in on synthetic GARCH prices; switch it to P4's
  data once `data/` and `backtest/` are ready.
- Headline figure: GRU advantage vs each asset's volatility persistence.
