"""Exercise the evaluator/reward interface used by a terminal-schedule RL agent.

This is an integration example, not a trained policy. Run from any directory.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluator import Evaluator


def demo():
    # Explicit fixture preserves the earlier demo without requiring a generator.
    problem = dict(widths=[256] * 9, heights=[2048] * 9,
                   inputs=[[i] for i in range(8)], outputs=[[i + 1] for i in range(8)],
                   base_costs=[500, 500, 200, 200, 500, 1000, 500, 1000],
                   op_types=["Pointwise"] * 8, fast_memory_capacity=600000,
                   slow_memory_bandwidth=10, native_granularity=[128, 128])
    with Evaluator(problem) as ev:
        baseline = ev.baseline()["schedule"]
        baseline.pop("subgraph_latencies")
        fused = dict(subgraphs=[list(range(8))],
                     granularities=[baseline["granularities"][0]], tensors_to_retain=[[]])
        invalid = dict(baseline, subgraphs=list(reversed(baseline["subgraphs"])))
        results = {name: ev.reward(schedule) for name, schedule in
                   (("baseline", baseline), ("fused", fused), ("invalid_order", invalid))}
        if (not results["baseline"]["evaluation"]["valid"]
            or not results["fused"]["evaluation"]["valid"]
            or results["baseline"]["reward"] != 0.5
            or results["fused"]["reward"] < 0.5
            or results["invalid_order"]["evaluation"]["valid"]
            or results["invalid_order"]["reward"] != -1):
            raise RuntimeError("Evaluator/reward integration check failed")
        return dict(configuration=ev.configuration, problem=problem,
                    schedules=dict(baseline=baseline, fused=fused, invalid_order=invalid), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/rl-evaluator-demo.json")
    args = parser.parse_args()
    report = demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {name: dict(valid=row["evaluation"]["valid"], reward=row["reward"],
                          modeled_latency=row["evaluation"]["total_latency"])
               for name, row in report["results"].items()}
    print(json.dumps(summary, indent=2))
    print(f"Saved problem, schedules, model configuration, and results to {args.output}")


if __name__ == "__main__":
    main()
