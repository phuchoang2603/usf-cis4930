import argparse
import csv
import hashlib
import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

from .workloads import (
    FAMILIES, MIN_NODES, GenerationConfig, augment, generate, statistics, topology_id, validate,
)


def _integers(value):
    try:
        values = tuple(int(x) for x in value.split(","))
        if not values or any(x <= 0 for x in values):
            raise ValueError
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error


def _families(value):
    values = tuple(value.split(",")) if value else ()
    if len(set(values)) != len(values) or any(x not in FAMILIES for x in values):
        raise argparse.ArgumentTypeError(f"choose distinct families from {','.join(FAMILIES)}")
    return values


def _digest(problem):
    return hashlib.sha256(json.dumps(problem, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _paths(path):
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"input path does not exist: {path}")
    if (path / "manifest.json").is_file():
        manifest = json.loads((path / "manifest.json").read_text())
        paths = [(path / row["path"]).resolve() for row in manifest["workloads"]]
        if any(not p.is_relative_to(path.resolve()) for p in paths):
            raise ValueError("manifest workload path escapes the dataset directory")
    else:
        paths = sorted(path.rglob("*.json"))
    if not paths:
        raise ValueError(f"no workload JSON files found in {path}")
    return paths


def _write_dataset(output, records, metadata):
    if output.exists():
        raise ValueError(f"output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".synthetic-", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        root.mkdir()
        manifest_rows, csv_rows = [], []
        split_topologies = {}
        hashes = set()
        for index, (problem, info) in enumerate(records):
            content_hash = _digest(problem)
            if content_hash in hashes:
                raise ValueError("duplicate workload generated; increase parameter diversity or change seed")
            hashes.add(content_hash)
            topology = topology_id(problem)
            previous_split = split_topologies.setdefault(topology, info["split"])
            if previous_split != info["split"]:
                raise ValueError("same topology assigned to different splits")
            relative = Path(info["split"]) / f'{index:05d}_{info["family"]}.json'
            target = root / relative
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps(problem, indent=2) + "\n")
            row = dict(info, path=relative.as_posix(), sha256=content_hash, topology_id=topology)
            manifest_rows.append(row)
            csv_rows.append(dict(path=relative.as_posix(), split=info["split"], family=info["family"],
                                 **statistics(problem)))
        manifest = dict(
            format_version=1, **metadata, workload_count=len(records),
            memory_units="tensor elements, as in PROBLEM.md; no byte-size multiplier",
            validation="schema, shapes, DAG, and minimum 1x1x1 per-operation working set",
            evaluator_validated=False,
            workloads=manifest_rows,
        )
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (root / "statistics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        root.rename(output)
    counts = {split: sum(row["split"] == split for row in manifest_rows)
              for split in ("train", "validation", "test")}
    print(f"Wrote {len(records)} workloads to {output}: {counts}")
    print("Schema, shapes, DAGs, and minimum working sets checked. Latency evaluator not run.")


def _generate(args):
    if args.count < 1 or not args.families:
        raise ValueError("count and number of selected families must be positive")
    if args.min_nodes > args.max_nodes or args.min_nodes < max(MIN_NODES[f] for f in args.families):
        raise ValueError("invalid node range for the selected families")
    validation, test = set(args.validation_families), set(args.test_families)
    if validation & test:
        raise ValueError("validation and test families must be disjoint")
    if (validation | test) - set(args.families):
        raise ValueError("held-out families must be included in --families (use an empty string to disable)")
    config = GenerationConfig(
        dimensions=args.dimensions, memory_capacities=args.memory_capacities,
        bandwidths=args.bandwidths, native_sizes=args.native_sizes,
        pointwise_costs=args.pointwise_costs, matmul_costs=args.matmul_costs,
        matmul_probability=args.matmul_probability,
    )
    rng = random.Random(args.seed)
    records = []
    for index in range(args.count):
        family = args.families[index % len(args.families)]
        nodes = rng.randint(args.min_nodes, args.max_nodes)
        seed = rng.getrandbits(64)
        split = "validation" if family in validation else "test" if family in test else "train"
        records.append((generate(family, nodes, seed, config),
                        dict(family=family, split=split, seed=seed)))
    _write_dataset(args.output, records, dict(
        mode="synthetic", seed=args.seed, config=asdict(config),
        node_range=[args.min_nodes, args.max_nodes], families=args.families,
        split_strategy="held-out graph families",
        validation_families=args.validation_families, test_families=args.test_families,
        official_benchmarks_used=False,
    ))


def _augment(args):
    if args.variants < 1:
        raise ValueError("variants must be positive")
    rng = random.Random(args.seed)
    records = []
    for source in _paths(args.source):
        original = json.loads(source.read_text())
        source_hash = _digest(original)
        group = topology_id(original)
        bucket = int(hashlib.sha256(f"{args.seed}:{group}".encode()).hexdigest(), 16) % 10
        split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        for _ in range(args.variants):
            seed = rng.getrandbits(64)
            scales = dict(dimension_scale=rng.choice(args.dimension_scales),
                          cost_scale=rng.uniform(0.7, 1.4), memory_scale=rng.uniform(0.5, 1.5),
                          bandwidth_scale=rng.uniform(0.5, 1.5))
            problem = augment(original, seed=seed, **scales)
            records.append((problem, dict(family="augmented", split=split, seed=seed,
                                         source=str(source), source_sha256=source_hash,
                                         source_topology_id=group, **scales)))
    _write_dataset(args.output, records, dict(
        mode="augmentation", seed=args.seed, variants_per_source=args.variants,
        split_strategy="80/10/10 hash buckets by source topology; small sets may have empty splits",
        holdout_caveat="Source benchmarks and their variants are not independent test graphs. "
                       "Exclude training sources from any real-benchmark transfer evaluation.",
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and validate synthetic MLSys scheduling workloads (Python standard library only)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate", help="generate fresh DAGs; official benchmarks stay untouched")
    generate_parser.add_argument("--output", type=Path, required=True, help="new dataset directory")
    generate_parser.add_argument("--count", type=int, default=100)
    generate_parser.add_argument("--seed", type=int, default=4930)
    generate_parser.add_argument("--min-nodes", type=int, default=8)
    generate_parser.add_argument("--max-nodes", type=int, default=152)
    generate_parser.add_argument("--families", type=_families, default=FAMILIES,
                                 help="comma-separated chain,branch,diamond,matmul,mixed")
    generate_parser.add_argument("--validation-families", type=_families, default=("branch",))
    generate_parser.add_argument("--test-families", type=_families, default=("mixed",))
    defaults = GenerationConfig()
    for name in ("dimensions", "memory_capacities", "bandwidths", "native_sizes",
                 "pointwise_costs", "matmul_costs"):
        generate_parser.add_argument("--" + name.replace("_", "-"), type=_integers,
                                     default=getattr(defaults, name), help="comma-separated positive integers")
    generate_parser.add_argument("--matmul-probability", type=float, default=0.45,
                                 help="MatMul probability within matmul/mixed regions after required motifs")
    generate_parser.set_defaults(run=_generate)
    augment_parser = commands.add_parser(
        "augment", help="perturb shapes/costs/hardware; keep source graphs out of independent transfer tests"
    )
    augment_parser.add_argument("--source", type=Path, required=True, help="workload JSON or directory")
    augment_parser.add_argument("--output", type=Path, required=True, help="new dataset directory")
    augment_parser.add_argument("--variants", type=int, default=5)
    augment_parser.add_argument("--dimension-scales", type=_integers, default=(1, 2))
    augment_parser.add_argument("--seed", type=int, default=4930)
    augment_parser.set_defaults(run=_augment)
    validate_parser = commands.add_parser("validate", help="check a JSON file or a workload/dataset directory")
    validate_parser.add_argument("source", type=Path)
    validate_parser.set_defaults(run=_validate)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        parser.exit(2, f"error: {error}\n")


def _validate(args):
    paths = _paths(args.source)
    for path in paths:
        try:
            validate(json.loads(path.read_text()))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}: {error}") from error
    print(f"Validated {len(paths)} workloads: schema, tensor semantics, DAG, minimum working set.")
    print("This does not validate a candidate schedule or its latency.")


if __name__ == "__main__":
    main()
