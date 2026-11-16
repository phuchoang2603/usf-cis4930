"""Compare original Huawei/VNRC CLI and project wrapper on identical raster schedules."""
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluator import Evaluator, MODEL_ID
from tools.assess_vnrc import explicit_raster


def stats(samples):
    ordered = sorted(samples)
    return dict(count=len(samples), median=statistics.median(samples),
                p95=ordered[math.ceil(len(ordered) * 0.95) - 1], max=max(samples))


def cli(problem, schedule):
    start = time.perf_counter_ns()
    result = subprocess.run([str(ROOT / "build/vnrc/evaluate"), str(problem), str(schedule)],
                            capture_output=True, text=True, timeout=20)
    ms = (time.perf_counter_ns() - start) / 1e6
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return ms, json.loads(result.stdout)


def main():
    output = ROOT / "build/evaluator-comparison"
    output.mkdir(parents=True, exist_ok=True)
    contexts, cases = [], []
    try:
        for path in sorted((ROOT / "benchmarks").glob("*.json")):
            problem = json.loads(path.read_text())
            ev = Evaluator(problem)
            contexts.append(ev)
            schedule = ev.baseline()["schedule"]
            schedule["traversal_orders"] = [None] * len(schedule["subgraphs"])
            schedule = explicit_raster(problem, schedule)
            target = output / f"{path.stem}.solution.json"
            target.write_text(json.dumps(schedule) + "\n")
            _, original = cli(path, target)
            wrapped = ev.evaluate(schedule, submission=True)
            if not wrapped["valid"] or not math.isclose(original["our_total"], wrapped["total_latency"], abs_tol=0.011):
                raise RuntimeError(f"Score mismatch on {path}")
            cases.append(dict(source=str(path.relative_to(ROOT)), nodes=len(problem["op_types"]),
                              original_total=original["our_total"], wrapper_total=wrapped["total_latency"],
                              scores_match_within_stdout_rounding=True, original_ms=[], wrapper_ms=[],
                              schedule=schedule, target=target, path=path))
        rng = random.Random(4930)
        for _ in range(9):
            order = list(range(len(cases)))
            rng.shuffle(order)
            for i in order:
                case = cases[i]
                # Alternate the order of the two approaches to reduce order bias.
                def original_call():
                    elapsed, result = cli(case["path"], case["target"])
                    case["original_ms"].append(elapsed)
                    if not math.isclose(result["our_total"], case["original_total"], abs_tol=0.001):
                        raise RuntimeError("unstable CLI score")
                def wrapper_call():
                    start = time.perf_counter_ns()
                    result = contexts[i].evaluate(case["schedule"], submission=True)
                    case["wrapper_ms"].append((time.perf_counter_ns() - start) / 1e6)
                    if not result["valid"] or result["total_latency"] != case["wrapper_total"]:
                        raise RuntimeError("unstable wrapper score")
                if rng.randrange(2):
                    original_call()
                    wrapper_call()
                else:
                    wrapper_call()
                    original_call()
        summary = dict(original_cli_ms=stats([v for c in cases for v in c["original_ms"]]),
                       wrapper_api_ms=stats([v for c in cases for v in c["wrapper_ms"]]))
        summary["median_time_ratio"] = summary["original_cli_ms"]["median"] / summary["wrapper_api_ms"]["median"]
        for case in cases:
            for key in ("schedule", "path", "target"):
                del case[key]
            case["original_stats_ms"] = stats(case["original_ms"])
            case["wrapper_stats_ms"] = stats(case["wrapper_ms"])
        report = dict(date="2026-10-30", model_id=MODEL_ID, schedules=len(cases),
                      scores_matched=sum(c["scores_match_within_stdout_rounding"] for c in cases),
                      method="24 official benchmark inputs; identical deterministic no-fusion schedules with explicit raster permutations and correct latency labels. 9 shuffled rounds, one warm-up per approach; randomized approach order. CLI includes process startup, IO, diagnostics and best_cost search; API reuses its problem context and includes validation, permutation/label checks and native scoring. Normal desktop background load uncontrolled.",
                      summary=summary, cases=cases)
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Scores agree: {report['scores_matched']}/{report['schedules']}")
    finally:
        for ev in contexts:
            ev.close()


if __name__ == "__main__":
    main()
