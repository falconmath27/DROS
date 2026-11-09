"""Run several experiments at the same time, one process each, then merge the results.
Run from the project root:
    python -m experiments.run_parallel                 # E1-E5, 2 threads per process
    python -m experiments.run_parallel E3 E4 --threads 3

Each experiment writes results/raw_<E>.jsonl and results/logs/<E>.log while it runs.
At the end the per-experiment files are merged (in E1..E5 order) into results/raw.jsonl,
the logs into results/run_log.txt, and the summary, tables and paired comparisons are
regenerated. An experiment that fails keeps its log; the others are merged anyway.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", default=["E1", "E2", "E3", "E4", "E5"])
    ap.add_argument("--threads", type=int, default=2, help="torch threads per process")
    ap.add_argument("--max-procs", type=int, default=5,
                    help="at most this many experiments at once; the rest wait their turn "
                         "(in the order given), which limits memory")
    ap.add_argument("--merge-only", action="store_true",
                    help="do not run anything; merge existing raw_<E>.jsonl files and logs")
    ap.add_argument("--no-merge", action="store_true",
                    help="leave raw_<E>.jsonl files in place (e.g. while another experiment "
                         "is still running); merge later with --merge-only")
    args = ap.parse_args()
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True)
    if args.merge_only:
        return merge()

    def start(e):
        part = os.path.join(RES, f"raw_{e}.jsonl")
        if os.path.exists(part):
            os.remove(part)
        env = dict(os.environ, RESULTS_FILE=part,
                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   PYTHONUNBUFFERED="1")
        log = open(os.path.join(RES, "logs", f"{e}.log"), "w")
        p = subprocess.Popen([sys.executable, "-m", "experiments.run_all", e], cwd=ROOT,
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"started {e}", flush=True)
        return p, log, time.time()

    waiting, running = list(args.experiments), {}
    while waiting or running:
        while waiting and len(running) < args.max_procs:
            e = waiting.pop(0)
            running[e] = start(e)
        time.sleep(5)
        for e, (p, log, t0) in list(running.items()):
            if p.poll() is not None:
                log.close()
                print(f"{e} finished with exit code {p.returncode} after "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
                del running[e]
    if not args.no_merge:
        merge()


def merge():
    """Rebuild results/raw.jsonl: for each experiment, rows from a fresh raw_<E>.jsonl if
    there is one, otherwise the rows already in raw.jsonl. Then rebuild the run log,
    summary, tables and paired comparisons."""
    import json
    order = ["E1", "E2", "E3", "E4", "E5"]
    old = []
    main_file = os.path.join(RES, "raw.jsonl")
    if os.path.exists(main_file):
        with open(main_file) as f:
            old = [l for l in f if l.strip()]
    rows = []
    for e in order:
        part = os.path.join(RES, f"raw_{e}.jsonl")
        if os.path.exists(part):
            with open(part) as f:
                rows += [l for l in f if l.strip()]
        else:
            rows += [l for l in old if json.loads(l)["experiment"] == e]
    with open(main_file, "w") as out:
        out.writelines(rows)
    for e in order:
        part = os.path.join(RES, f"raw_{e}.jsonl")
        if os.path.exists(part):
            os.remove(part)
    with open(os.path.join(RES, "run_log.txt"), "w") as out:
        for e in order:
            path = os.path.join(RES, "logs", f"{e}.log")
            if os.path.exists(path):
                with open(path) as f:
                    out.write(f.read())

    from experiments import paired, summarize
    summarize.main()
    paired.main()


if __name__ == "__main__":
    main()
