import os
import argparse
import csv
import numpy as np

os.environ.setdefault("RESULTS_FILE", "results/fbm_raw.jsonl")

from simulators.fbm import fbm_stopping_problem
from experiments.runner import run
from baselines.dos import DOS
from baselines.fqi import FQI
from ospg import OSPG


H_VALUES = [
    0.05, 0.10, 0.20, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 0.95
]

METHODS = [
    "DOS",
    "FQI-MLP",
    "FQI-GRU",
    "OSPG-MLP",
    "OSPG-GRU"
]


def smoke_overrides(seed):
    """Reduced training settings for testing the experiment pipeline."""

    return {
        "DOS": lambda: DOS(
            itm_only=False, steps=300, batch=128, seed=seed
        ),
        "FQI-MLP": lambda: FQI(
            kind="mlp", itm_only=False, steps=300,
            batch=128, seed=seed, eval_every=50
        ),
        "FQI-GRU": lambda: FQI(
            kind="gru", itm_only=False, steps=300,
            batch=128, seed=seed, eval_every=50
        ),
        "OSPG-MLP": lambda: OSPG(
            kind="mlp", itm_only=False, steps=300,
            batch=128, seed=seed, eval_every=50, min_steps=300
        ),
        "OSPG-GRU": lambda: OSPG(
            kind="gru", itm_only=False, steps=300,
            batch=128, seed=seed, eval_every=50, min_steps=300
        )
    }

def medium_overrides(seed):
    return {
        "DOS": lambda: DOS(
            itm_only=False, steps=500, batch=128, seed=seed
        ),
        "FQI-MLP": lambda: FQI(
            kind="mlp", itm_only=False, steps=1500,
            batch=256, seed=seed
        ),
        "FQI-GRU": lambda: FQI(
            kind="gru", itm_only=False, steps=1500,
            batch=256, seed=seed
        ),
        "OSPG-MLP": lambda: OSPG(
            kind="mlp", itm_only=False, steps=3000,
            batch=256, seed=seed, min_steps=2000
        ),
        "OSPG-GRU": lambda: OSPG(
            kind="gru", itm_only=False, steps=3000,
            batch=256, seed=seed, min_steps=2000
        )
    }


def save_summary(rows, filename):
    """Save mean and standard deviation across random splits."""

    groups = {}

    for row in rows:
        key = (row["problem"], row["method"])
        groups.setdefault(key, []).append(row["value"])

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hurst", "Method", "Mean reward", "Std reward"])

        for (problem, method), values in sorted(groups.items()):
            values = np.asarray(values)
            h = float(problem.split("=")[1])

            std = values.std(ddof=1) if len(values) > 1 else 0.0

            writer.writerow([
                h, method, values.mean(), std
            ])

    print(f"\nSummary saved to {filename}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--medium", action="store_true")
    args = parser.parse_args()

    if args.smoke and args.medium:
        parser.error("Choose either smoke or medium")

    if args.smoke:
        h_values = [0.30, 0.50]
        seeds = [0]
        N = 2000
        n_steps = 30
        experiment = "FBM_SMOKE"

    elif args.medium:
        h_values = H_VALUES
        seeds = range(2)
        N = 20_000
        n_steps = 100
        experiment = "FBM_MEDIUM"

    else:
        h_values = H_VALUES
        seeds = range(10)
        N = 40_000
        n_steps = 100
        experiment = "FBM"

    all_rows = []

    for h in h_values:

        print(f"\n========== H = {h} ==========", flush=True)

        problem = fbm_stopping_problem(
            H=h,
            n_steps=n_steps,
            N=N,
            seed=1000 + int(h * 100)
        )

        for seed in seeds:

            print(f"\n--- Split {seed + 1} ---", flush=True)

            splits = problem.random_split(seed=seed)

            if args.smoke:
                overrides = smoke_overrides(seed)
            elif args.medium:
                overrides = medium_overrides(seed)
            else:
                overrides = None

            rows = run(
                experiment,
                f"h={h:.2f}",
                splits,
                METHODS,
                seed,
                method_overrides=overrides
            )

            all_rows.extend(rows)

    os.makedirs("results", exist_ok=True)

    summary_file = (
        "results/fbm_smoke_summary.csv" if args.smoke
        else "results/fbm_medium_summary.csv" if args.medium
        else "results/fbm_summary.csv"
    )

    save_summary(all_rows, summary_file)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()