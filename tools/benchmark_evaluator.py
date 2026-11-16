"""Measure the complete reusable Python evaluator, including validation and C++ calls."""
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluator import Evaluator, MODEL_ID


def stats(values):
    values = sorted(values)
    return dict(count=len(values), median=statistics.median(values),
                p95=values[math.ceil(0.95 * len(values)) - 1], max=max(values))


def main():
    paths = sorted((ROOT / "benchmarks").glob("*.json"))
    paths += sorted((ROOT / "data/generated/synthetic").glob("*/*.json"))
    cases, contexts = [], []
    try:
        for path in paths:
            problem = json.loads(path.read_text())
            started = time.perf_counter_ns()
            ev = Evaluator(problem)
            setup_us = (time.perf_counter_ns() - started) / 1000
            contexts.append(ev)
            baseline = ev.baseline()
            schedule = baseline["schedule"]
            del schedule["subgraph_latencies"]  # An RL action need not predict its own reward.
            ev.evaluate(schedule)  # warm-up
            cases.append(dict(source=str(path.relative_to(ROOT)), nodes=len(problem["op_types"]),
                              group="official" if path.parent.name == "benchmarks" else "synthetic",
                              setup_us=setup_us, samples_us=[], reward_samples_us=[], schedule=schedule,
                              configuration=ev.configuration,
                              baseline_latency=baseline["evaluation"]["total_latency"]))
        order = list(range(len(cases)))
        rng = random.Random(4930)
        for _ in range(9):
            rng.shuffle(order)
            for i in order:
                start = time.perf_counter_ns()
                result = contexts[i].evaluate(cases[i]["schedule"])
                elapsed = (time.perf_counter_ns() - start) / 1000
                if not result["valid"] or not math.isclose(result["total_latency"], cases[i]["baseline_latency"]):
                    raise RuntimeError("inconsistent scoring")
                cases[i]["samples_us"].append(elapsed)
                start = time.perf_counter_ns()
                reward = contexts[i].reward(cases[i]["schedule"])
                reward_elapsed = (time.perf_counter_ns() - start) / 1000
                if reward["reward"] != 0.5:
                    raise RuntimeError("inconsistent baseline reward")
                cases[i]["reward_samples_us"].append(reward_elapsed)
        for case in cases:
            del case["schedule"]
            case["timing_us"] = stats(case["samples_us"])
            case["reward_timing_us"] = stats(case["reward_samples_us"])
        groups = {group: stats([sample for case in cases if case["group"] == group
                                for sample in case["samples_us"]]) for group in ("official", "synthetic")}
        reward_groups = {group: stats([sample for case in cases if case["group"] == group
                                       for sample in case["reward_samples_us"]]) for group in ("official", "synthetic")}
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        report = dict(date="2026-10-30", model_id=MODEL_ID, platform=platform.platform(),
                      backend_sha256=hashlib.sha256((ROOT / f"build/evaluator/libevaluator_backend{suffix}").read_bytes()).hexdigest(),
                      method="9 sequential randomized rounds after one warm-up per schedule. Reused problem contexts; complete Python validation, schedule serialization, native subgraph construction and cost scoring, and result parsing included. Each evaluate() call is followed by reward() timing with an already cached baseline. Initialization and baseline generation excluded. Wall time in microseconds; desktop background load uncontrolled.",
                      groups=groups, reward_groups=reward_groups, cases=cases)
        target = ROOT / "build/evaluator-wrapper-results.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(groups, indent=2))
        print("Reward API (microseconds):", json.dumps(reward_groups, indent=2))
        print(f"Saved {target}")
    finally:
        for context in contexts:
            context.close()


if __name__ == "__main__":
    main()
