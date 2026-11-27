import argparse
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from evaluator import Evaluator, MODEL_ID

from .synthetic_graphs import MOTIFS, generate_problem


PROFILES = (
    (range(4, 8), (25000, 40000, 60000), (10, 15, 20)),
    (range(8, 13), (60000, 120000, 250000), (15, 25, 40)),
    (range(13, 21), (150000, 350000, 600000), (30, 60, 100)),
)


def _decode_topology(index, depths):
    base = len(MOTIFS)
    for depth in depths:
        if index < base**depth:
            digits = []
            for _ in range(depth):
                index, digit = divmod(index, base)
                digits.append(MOTIFS[digit])
            return digits
        index -= base**depth
    raise ValueError("topology index exceeds the available sequences")


def _sample_topology_indices(rng, counts):
    pools = []
    for offset, (depths, _, _) in enumerate(PROFILES):
        needed = sum(len(range(offset, count, len(PROFILES))) for count in counts)
        space = sum(len(MOTIFS)**depth for depth in depths)
        if needed > space:
            raise ValueError(
                f"requested {needed} distinct graphs of depth {depths.start}–{depths.stop - 1}; only {space} exist"
            )
        pools.append(iter(rng.sample(range(space), needed)))
    return pools


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_dataset(output, seed, counts):
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    if any(count < 0 for count in counts.values()) or not sum(counts.values()):
        raise ValueError("split counts must be nonnegative and at least one must be positive")
    rng = Random(seed)
    pools = _sample_topology_indices(rng, counts.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".synthetic-", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        root.mkdir()
        splits = {}
        for split, count in counts.items():
            (root / split).mkdir()
            rows = splits[split] = []
            for index in range(count):
                profile = index % len(PROFILES)
                depths, capacities, bandwidths = PROFILES[profile]
                motifs = _decode_topology(next(pools[profile]), depths)
                problem = generate_problem(motifs, rng, rng.choice(capacities), rng.choice(bandwidths))
                path = f"{split}/{index:05d}.json"
                try:
                    with Evaluator(problem) as evaluator:
                        evaluator.baseline()
                except (ValueError, RuntimeError) as error:
                    raise ValueError(f"{path}: {error}") from error
                _write_json(root / path, problem)
                rows.append(dict(path=path, motifs=[motif.__name__ for motif in motifs]))
        _write_json(root / "manifest.json", dict(seed=seed, evaluator=MODEL_ID, splits=splits))
        root.rename(output)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline-validated MLSys graph scheduling workloads.")
    parser.add_argument("--output", type=Path, required=True, help="new dataset directory")
    parser.add_argument("--seed", type=int, default=4930)
    for split, default in (("train", 80), ("validation", 10), ("test", 10)):
        parser.add_argument(f"--{split}", type=int, default=default, help=f"workload count (default: {default})")
    args = parser.parse_args()
    counts = dict(train=args.train, validation=args.validation, test=args.test)
    try:
        write_dataset(args.output, args.seed, counts)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {sum(counts.values())} workloads to {args.output}")


if __name__ == "__main__":
    main()
