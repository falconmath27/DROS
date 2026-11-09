"""Paired comparison of methods on the same test paths (support for P5).

For every (experiment, problem) it compares each method with a reference method on the
SAME test paths: per seed, the mean of the per-path difference and its standard error;
then the average over seeds, with standard error sqrt(sum se_s^2) / n_seeds.
Pairing removes the path-to-path noise shared by both methods, so it detects much smaller
differences than comparing the two means and their separate standard errors.

The standard errors assume independent test paths. That holds for the simulated option
problems (E1-E4); the market windows of E5 overlap in time, so their errors here are too
small and P5's block bootstrap (resampling by date) should be used instead.

Run from the project root:  python -m experiments.paired   -> results/paired.md
"""
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.join(os.path.dirname(__file__), "..", "results")
PATHS = os.path.join(HERE, "paths")
PREFERRED_REFERENCE = ["OSPG-GRU", "OSPG-MLP"]


def load():
    """{(experiment, problem): {seed: {method: reward array}}}"""
    out = defaultdict(dict)
    for name in sorted(os.listdir(PATHS)):
        m = re.match(r"(E\d+)_(.+)_seed(\d+)\.npz$", name)
        if not m:
            continue
        exp, prob, seed = m.group(1), m.group(2), int(m.group(3))
        z = np.load(os.path.join(PATHS, name))
        out[(exp, prob)][seed] = {k[len("reward_"):]: z[k] for k in z.files if k.startswith("reward_")}
    return out


def compare(by_seed, ref):
    rows = []
    methods = sorted({m for d in by_seed.values() for m in d} - {ref})
    for m in methods:
        diffs, ses = [], []
        for seed, d in sorted(by_seed.items()):
            if m in d and ref in d and len(d[m]) == len(d[ref]):
                x = d[ref].astype(float) - d[m].astype(float)
                diffs.append(x.mean())
                ses.append(x.std(ddof=1) / np.sqrt(len(x)))
        if diffs:
            diff = float(np.mean(diffs))
            se = float(np.sqrt(np.sum(np.square(ses))) / len(ses))
            rows.append(dict(method=m, seeds=len(diffs), diff=diff, se=se, z=diff / se if se > 0 else np.nan))
    return rows


def main():
    data = load()
    lines = ["# Paired comparisons on identical test paths", "",
             "diff = (reference method) - (other method), averaged over seeds; se = paired "
             "standard error; z = diff / se. |z| > 2 is a difference unlikely to be noise. "
             "E5 rows assume independent windows, which overlapping windows are not: use P5's "
             "block bootstrap there.", ""]
    for (exp, prob), by_seed in sorted(data.items()):
        methods = {m for d in by_seed.values() for m in d}
        ref = next((r for r in PREFERRED_REFERENCE if r in methods), None)
        if ref is None:
            continue
        lines += [f"## {exp} - {prob} (reference: {ref})", "",
                  "| other method | seeds | diff | se | z |", "| --- | --- | --- | --- | --- |"]
        for r in compare(by_seed, ref):
            lines.append(f"| {r['method']} | {r['seeds']} | {r['diff']:+.4f} | {r['se']:.4f} | {r['z']:+.1f} |")
        lines.append("")
    with open(os.path.join(HERE, "paired.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
