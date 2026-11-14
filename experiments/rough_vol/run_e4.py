import os
os.environ["RESULTS_FILE"] = "results/e4_h_sweep_raw.jsonl"

from simulators.rbergomi import rbergomi_put_problem
from experiments.runner import run
from baselines.dos import DOS
from ospg import OSPG


H_VALUES = [0.07, 0.10, 0.30, 0.50]


def main():

    for H in H_VALUES:

        print(f"\n========== H = {H} ==========", flush=True)

        seed = 42

        problem = rbergomi_put_problem(
            N=3000,
            n_steps=25,
            exercise_every=1,
            H=H,
            seed=seed
        )

        splits = problem.random_split(seed=0)

        overrides = {
            "DOS": lambda: DOS(
                itm_only=True, steps=150,
                batch=512, seed=seed
            ),

            "OSPG-MLP": lambda: OSPG(
                kind="mlp", itm_only=True,
                steps=500, batch=256,
                seed=seed, eval_every=100,
                min_steps=500
            ),

            "OSPG-GRU": lambda: OSPG(
                kind="gru", itm_only=True,
                steps=500, batch=256,
                seed=seed, eval_every=100,
                min_steps=500
            )
        }

        run(
            "E4_H_SWEEP",
            f"rbergomi_H={H}",
            splits,
            ["LSM", "DOS", "OSPG-MLP", "OSPG-GRU"],
            seed=seed,
            method_overrides=overrides
        )


if __name__ == "__main__":
    main()