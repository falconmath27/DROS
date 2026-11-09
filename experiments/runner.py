"""Runs a set of methods on train/val/test splits and appends one JSON line per result to
results/raw.jsonl (or the file named by the RESULTS_FILE environment variable), so partial
results survive an interrupted run.

For P5's paired tests, the per-path test rewards and stopping dates of every run are also
saved to results/paths/<experiment>_<problem>_seed<seed>.npz, one array per method
(reward_<method>, tau_<method>). Within one file all methods were evaluated on the SAME
test paths, so differences between methods can be compared path by path."""
import json
import os
import re
import time

import numpy as np

from baselines.dos import DOS
from baselines.fqi import FQI
from baselines.lsm import LSM
from common.problem import evaluate
from ospg.train import OSPG

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS = os.environ.get("RESULTS_FILE") or os.path.join(RESULTS_DIR, "raw.jsonl")
PATHS_DIR = os.path.join(RESULTS_DIR, "paths")


def save_paths(experiment, problem_label, seed, method, reward, tau):
    """Add one method's per-path results to the .npz file of this (experiment, problem, seed)."""
    os.makedirs(PATHS_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.=-]", "_", f"{experiment}_{problem_label}_seed{seed}")
    path = os.path.join(PATHS_DIR, safe + ".npz")
    arrays = dict(np.load(path)) if os.path.exists(path) else {}
    arrays[f"reward_{method}"] = reward.astype(np.float32)
    arrays[f"tau_{method}"] = tau.astype(np.int16)
    np.savez_compressed(path, **arrays)


class HoldToEnd:
    """Never stop early (the 'hold' rule for trading; the European-style rule for options)."""

    def fit(self, X, R, X_val=None, R_val=None):
        return self

    def stop(self, X, R):
        return np.full(R.shape[0], R.shape[1] - 1)


def method_factory(name, itm_only, seed):
    table = {
        "Hold": lambda: HoldToEnd(),
        "LSM": lambda: LSM(itm_only=itm_only),
        "DOS": lambda: DOS(itm_only=itm_only, seed=seed),
        "FQI-MLP": lambda: FQI(kind="mlp", itm_only=itm_only, seed=seed),
        "FQI-GRU": lambda: FQI(kind="gru", itm_only=itm_only, seed=seed),
        "OSPG-MLP": lambda: OSPG(kind="mlp", itm_only=itm_only, seed=seed),
        "OSPG-GRU": lambda: OSPG(kind="gru", itm_only=itm_only, seed=seed),
    }
    return table[name]()


def run(experiment, problem_label, splits, methods, seed, extra=None, method_overrides=None):
    """splits = (train, val, test) Problems. Appends results and returns them."""
    train, val, test = splits
    rows = []
    only = os.environ.get("ONLY_METHODS")          # e.g. ONLY_METHODS=OSPG-MLP,OSPG-GRU
    if only:
        methods = [m for m in methods if m in only.split(",")]
    for name in methods:
        if method_overrides and name in method_overrides:
            m = method_overrides[name]()
        else:
            m = method_factory(name, train.itm_only, seed)
        t0 = time.time()
        m.fit(train.X, train.R, val.X, val.R)
        tau = m.stop(test.X, test.R)
        got = test.R[np.arange(test.N), tau]
        row = dict(experiment=experiment, problem=problem_label, method=name, seed=seed,
                   value=float(got.mean()), stderr=float(got.std() / np.sqrt(test.N)),
                   mean_tau=float(tau.mean()), n_train=train.N, n_test=test.N,
                   seconds=round(time.time() - t0, 1))
        if extra:
            row.update(extra)
        rows.append(row)
        save_paths(experiment, problem_label, seed, name, got, tau)
        print(json.dumps(row), flush=True)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row) + "\n")
    return rows
