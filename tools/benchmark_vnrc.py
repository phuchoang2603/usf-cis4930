"""Time unchanged VNRC CLI and prepared core scoring on assessment fixtures.

Run tools/assess_vnrc.py first. This measures wall time, not simulated latency.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def distribution(values):
    values = sorted(values)
    return dict(count=len(values), min=min(values), median=statistics.median(values),
                p95=values[max(0, math.ceil(0.95 * len(values)) - 1)], max=max(values),
                mean=statistics.mean(values))


def run_cli(binary, problem, solution):
    start = time.perf_counter_ns()
    p = subprocess.run([str(binary), str(problem), str(solution)],
                       capture_output=True, timeout=20)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    if p.returncode:
        raise RuntimeError(f"Evaluator failed for {problem}: {p.stderr.decode()[-500:]}")
    return elapsed_ms, json.loads(p.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "build/vnrc-assessment")
    parser.add_argument("--output", type=Path, default=ROOT / "build/vnrc-timing")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("use at least three repeats")
    fixtures, output = args.fixtures.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assessment = json.loads((fixtures / "summary.json").read_text())
    cli = ROOT / "build/vnrc/evaluate"
    driver = ROOT / "build/vnrc_timing_driver"
    compiler = "clang++" if platform.system() == "Darwin" else "c++"
    subprocess.run([compiler, "-std=c++20", "-O2", "-UNDEBUG", "-pthread",
                    f"-I{ROOT / 'external/vnrc/src'}", f"-I{ROOT / 'external/deps'}",
                    str(ROOT / "tools/vnrc_timing_driver.cpp"),
                    str(ROOT / "build/vnrc/libsolver_lib.a"), "-o", str(driver)], check=True)
    candidates = [("official_examples", r) for r in assessment["examples"]]
    candidates += [("official_benchmarks" if r["source"].startswith("benchmarks/")
                    else "synthetic", r) for r in assessment["corpus"]]
    candidates += [("compiler_archive", r) for r in assessment.get("archive", []) if r["accepted"]]
    rows = []
    for group, case in candidates:
        directory = fixtures / case["name"]
        problem, solution = directory / "problem.json", directory / "solution.json"
        _, result = run_cli(cli, problem, solution)  # one warm-up per input
        p = json.loads(problem.read_text())
        rows.append(dict(group=group, case=case["name"], source=case.get("source"),
                         nodes=len(p["op_types"]), steps=result["num_steps"],
                         expected_vnrc_total=result["our_total"], cli_ms_samples=[]))
    order = list(range(len(rows)))
    rng = random.Random(4930)
    for trial in range(args.repeats):
        rng.shuffle(order)
        for index in order:
            row = rows[index]
            directory = fixtures / row["case"]
            elapsed, result = run_cli(cli, directory / "problem.json", directory / "solution.json")
            if not math.isclose(result["our_total"], row["expected_vnrc_total"], abs_tol=0.01):
                raise RuntimeError("Evaluator score changed across identical invocations")
            row["cli_ms_samples"].append(elapsed)
        print(f"CLI round {trial + 1}/{args.repeats}: {len(rows)} schedules", flush=True)
    for row in rows:
        directory = fixtures / row["case"]
        result = subprocess.run([str(driver), str(directory / "problem.json"),
                                 str(directory / "solution.json")], capture_output=True,
                                text=True, timeout=20, check=True)
        core = json.loads(result.stdout)
        if not math.isclose(core["total"], row["expected_vnrc_total"], abs_tol=0.011):
            raise RuntimeError(f"Prepared core disagrees with CLI: {row['case']}")
        row["cli_ms"] = distribution(row["cli_ms_samples"])
        row["prepared_core_us"] = distribution(core["prepared_score_us"])
        row["core_samples"] = core
    groups = {}
    for group in sorted({r["group"] for r in rows}):
        selected = [r for r in rows if r["group"] == group]
        groups[group] = dict(schedules=len(selected),
                            cli_ms=distribution([x for r in selected for x in r["cli_ms_samples"]]),
                            core_us_across_schedule_medians=distribution(
                                [r["prepared_core_us"]["median"] for r in selected]))
    report = dict(platform=platform.platform(), timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  vnrc_revision=assessment["vnrc_revision"],
                  cli_sha256=hashlib.sha256(cli.read_bytes()).hexdigest(),
                  compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                  repeats=args.repeats, groups=groups, rows=rows,
                  method=dict(cli="Sequential subprocess wall time with stdout/stderr captured; includes process startup, JSON IO, diagnostics, and best_cost searches. One warm-up per schedule; randomized round order.",
                              core="7 batches, each >=10ms or 100000 iterations; per-call time from batch duration. Prepared graph and schedule cost computation only; no full schedule validation.",
                              core_units="microseconds", cli_units="milliseconds",
                              caveat="Warm local desktop measurement; no cache flushing, CPU pinning, or background-load control. Corpus schedules are feasible singleton smoke fixtures, not competition submissions."))
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
