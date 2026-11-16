import argparse
import json
from pathlib import Path
import sys

from . import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Validate and score a schedule with the pinned project evaluator")
    parser.add_argument("problem", type=Path)
    parser.add_argument("schedule", type=Path, nargs="?")
    parser.add_argument("--allow-snake", action="store_true")
    parser.add_argument("--submission", action="store_true", help="require and verify reported per-step latencies")
    parser.add_argument("--reward", action="store_true", help="include finite normalized RL reward")
    args = parser.parse_args()
    try:
        problem = json.loads(args.problem.read_text())
        with Evaluator(problem, allow_snake=args.allow_snake) as evaluator:
            if args.schedule is None:
                result = evaluator.baseline()
                valid = result["evaluation"]["valid"]
            else:
                schedule = json.loads(args.schedule.read_text())
                result = evaluator.evaluate(schedule, submission=args.submission)
                valid = result["valid"]
                if args.reward:
                    result = dict(result, reward=evaluator.reward(schedule)["reward"] if valid else -1.0)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if valid else 2
    except (RuntimeError, MemoryError) as error:
        print(json.dumps(dict(valid=False, total_latency=None, code="backend_or_setup_error", error=str(error))))
        return 1
    except (OSError, ValueError, TypeError, OverflowError) as error:
        print(json.dumps(dict(valid=False, total_latency=None, code="input_or_setup_error", error=str(error))))
        return 2


if __name__ == "__main__":
    sys.exit(main())
