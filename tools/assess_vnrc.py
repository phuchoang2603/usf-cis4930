"""Assess the unmodified VNRC CLI. Findings are data, not expected test failures.

Exit 0 means the assessment completed, not that VNRC is safe as a reward oracle.
Optional --compare-archive needs NumPy to run the existing ZIP's simulator.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic_data.workloads import validate
from tools.setup_vnrc import REVISION


def solution(groups, granularities, latencies=None, traversal=None, retain=None):
    n = len(groups)
    return dict(subgraphs=groups, granularities=granularities,
                tensors_to_retain=retain if retain is not None else [[] for _ in groups],
                traversal_orders=traversal if traversal is not None else [None] * n,
                subgraph_latencies=latencies if latencies is not None else [0] * n)


def explicit_raster(problem, schedule):
    """Make null's documented meaning explicit, for supported equal-shape sinks."""
    result = copy.deepcopy(schedule)
    for i, group in enumerate(result["subgraphs"]):
        if result["traversal_orders"][i] is not None:
            continue
        consumed = {t for op in group for t in problem["inputs"][op]}
        sinks = [t for op in group for t in problem["outputs"][op] if t not in consumed]
        out = sinks[0]
        w, h, _ = result["granularities"][i]
        tiles = math.ceil(problem["widths"][out] / w) * math.ceil(problem["heights"][out] / h)
        result["traversal_orders"][i] = list(range(tiles))
    return result


class Assessment:
    def __init__(self, binary, output):
        self.binary, self.output = binary, output
        output.mkdir(parents=True, exist_ok=True)

    def evaluate(self, name, problem, schedule):
        directory = self.output / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename, value in (("problem.json", problem), ("solution.json", schedule)):
            (directory / filename).write_text(json.dumps(value, indent=2) + "\n")
        try:
            p = subprocess.run([str(self.binary), str(directory / "problem.json"),
                                str(directory / "solution.json")], capture_output=True,
                               text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return dict(name=name, accepted=False, timeout=True, result=None)
        (directory / "stdout.txt").write_text(p.stdout)
        (directory / "stderr.txt").write_text(p.stderr)
        try:
            result = json.loads(p.stdout)
        except ValueError:
            result = None
        return dict(name=name, accepted=p.returncode == 0, returncode=p.returncode,
                    result=result, non_json_stdout=bool(p.stdout.strip()) and result is None)

    def examples(self):
        text = (ROOT / "PROBLEM.md").read_text().split("## Examples", 1)[1]
        current, example, strategy, rows = None, "", "", []
        for match in re.finditer(r"^### (Example[^\n]+)|^#### (Strategy[^\n]+)|```json\s*\n(.*?)```",
                                 text, re.M | re.S):
            if match[1]:
                example = match[1]
            elif match[2]:
                strategy = match[2]
            else:
                if match[3].strip() == "NA":  # Example 5A explicitly has no valid schedule.
                    continue
                value = json.loads(re.sub(r"//[^\n]*", "", match[3]))
                if "widths" in value:
                    current = value
                elif "subgraphs" in value:
                    name = f"examples/{len(rows):02d}"
                    row = self.evaluate(name, current, value)
                    expected = sum(value["subgraph_latencies"])
                    actual = (row["result"] or {}).get("our_total")
                    row.update(label=f"{example} / {strategy}", expected=expected,
                               actual=actual, matches=row["accepted"] and actual is not None
                               and math.isclose(actual, expected, abs_tol=0.02))
                    if not row["matches"]:
                        normalized = self.evaluate(name + "_explicit_raster", current,
                                                   explicit_raster(current, value))
                        row["explicit_raster_result"] = normalized
                    rows.append(row)
        return rows

    def adversarial(self):
        p = dict(widths=[128] * 3, heights=[128] * 3, inputs=[[0], [1]],
                 outputs=[[1], [2]], base_costs=[1000, 100], op_types=["Pointwise"] * 2,
                 fast_memory_capacity=35000, slow_memory_bandwidth=10,
                 native_granularity=[128, 128])
        base = solution([[0], [1]], [[128, 128, 1]] * 2, [3276.8] * 2)
        cases = []
        def case(name, schedule, expected, problem=None):
            cases.append((name, problem or p, schedule, expected))
        case("valid_chain", base, True)
        bad = copy.deepcopy(base)
        bad["subgraphs"] = [[1], [0]]
        case("consumer_before_producer", bad, False)
        bad = copy.deepcopy(base)
        bad["tensors_to_retain"] = [[0], []]
        case("retain_graph_input", bad, False, dict(p, fast_memory_capacity=100000))
        bad = copy.deepcopy(base)
        bad["subgraph_latencies"] = [1, 1]
        case("false_reported_latencies", bad, False)
        bad = copy.deepcopy(base)
        del bad["subgraph_latencies"]
        case("missing_required_latencies", bad, False)
        bad = copy.deepcopy(base)
        bad["granularities"] = [[128, 128, 1]]
        case("mismatched_array_lengths", bad, False)
        case("uncovered_op", solution([[0]], [[128, 128, 1]], [3276.8]), False)
        small = dict(p, fast_memory_capacity=100)
        case("out_of_memory", base, False, small)
        bad = copy.deepcopy(base)
        bad["subgraphs"] = [[0, 0], [1]]
        case("duplicate_op_within_step", bad, False)
        mm = dict(widths=[128] * 3, heights=[128] * 3, inputs=[[0, 1]], outputs=[[2]],
                  base_costs=[1500], op_types=["MatMul"], fast_memory_capacity=25000,
                  slow_memory_bandwidth=10, native_granularity=[128, 128])
        case("duplicate_traversal_indices", solution([[0]], [[64, 64, 128]], [7096],
                                                   [[0, 0, 0, 0]]), False, mm)
        case("zero_tile_width", solution([[0]], [[0, 64, 128]], [7096],
                                         [[0, 1, 2, 3]]), False, mm)
        case("arbitrary_valid_traversal", solution([[0]], [[64, 64, 128]], [7644],
                                                  [[0, 3, 1, 2]]), True, mm)
        # Op0's T1 becomes ephemeral in {0,1}; op2 cannot then load it.
        fork = dict(p, widths=[128] * 4, heights=[128] * 4, inputs=[[0], [1], [1]],
                    outputs=[[1], [2], [3]], base_costs=[100] * 3, op_types=["Pointwise"] * 3)
        case("unavailable_ephemeral_tensor", solution([[0, 1], [2]], [[128, 128, 1]] * 2),
             False, fork)
        rows = []
        for name, problem, schedule, expected in cases:
            row = self.evaluate(f"adversarial/{name}", problem, schedule)
            row.update(expected_accept=expected, matches=row["accepted"] == expected)
            if name == "arbitrary_valid_traversal":
                # 0->3 changes both, 3->1 reuses RHS, 1->2 changes both.
                row["expected_total"] = 3 * 2048 + 1500
                row["matches_cost"] = math.isclose(
                    (row["result"] or {}).get("our_total", -1), 7644, abs_tol=0.02)
            rows.append(row)
        return rows

    def corpus(self):
        rows = []
        paths = sorted((ROOT / "benchmarks").glob("*.json"))
        paths += sorted((ROOT / "data/generated/synthetic").glob("*/*.json"))
        for path in paths:
            p = json.loads(path.read_text())
            order = validate(p)
            grans = []
            for i in order:
                out = p["outputs"][i][0]
                w, h = math.gcd(64, p["widths"][out]), math.gcd(64, p["heights"][out])
                mm = p["op_types"][i] == "MatMul"
                k = math.gcd(64, p["widths"][p["inputs"][i][0]]) if mm else 1
                while ((w * k + h * k + w * h) if mm else
                       (len(set(p["inputs"][i])) + 1) * w * h) > p["fast_memory_capacity"]:
                    if w >= h and w > 1:
                        w //= 2
                    elif h > 1:
                        h //= 2
                    elif k > 1:
                        k //= 2
                    else:
                        raise ValueError("No minimum tile fits")
                grans.append([w, h, k])
            s = solution([[i] for i in order], grans)
            name = f"corpus/{len(rows):03d}"
            row = self.evaluate(name, p, s)
            row["source"] = str(path.relative_to(ROOT))
            # Smoke test only: upstream ignores these deliberately zero labels.
            row["feasible_and_covered"] = row["accepted"] and all(
                (row["result"] or {}).get(k) is True for k in ("feasible", "coverage"))
            rows.append(row)
        return rows

    def cost_semantics(self):
        rows = []
        for reduction in (128, 256, 512):
            # High bandwidth isolates compute from IO. These expected values
            # describe the archive's native-tile cost convention, not official gold.
            p = dict(widths=[reduction, 128, 128], heights=[128, reduction, 128],
                     inputs=[[0, 1]], outputs=[[2]], base_costs=[2000], op_types=["MatMul"],
                     fast_memory_capacity=35000, slow_memory_bandwidth=1000000,
                     native_granularity=[128, 128])
            expected = 2000 * reduction / 128
            row = self.evaluate(f"cost_semantics/K{reduction}", p,
                                solution([[0]], [[128, 128, 64]], [expected], [[0]]))
            row["archive_native_tile_convention_total"] = expected
            rows.append(row)
        return rows

    def archive(self):
        # Execute the requested repository's existing simulator without modifying it.
        import types
        sim = types.ModuleType("assessment_archive_sim")
        rows = []
        with ZipFile(ROOT / "hlo2mlsys.zip") as z:
            exec(compile(z.read("hlo2mlsys/sim.py"), "hlo2mlsys.zip/sim.py", "exec"), sim.__dict__)
            for path in sorted(z.namelist()):
                if not path.endswith(".solution.json"):
                    continue
                stem = Path(path).name.removesuffix(".solution.json")
                p = json.loads(z.read(f"hlo2mlsys/examples/problems/{stem}.json"))
                s = json.loads(z.read(path))
                try:
                    validate(p)
                    strict_valid = True
                except ValueError:
                    strict_valid = False
                total, errors = sim.evaluate_solution(sim.Problem(p), s)
                row = self.evaluate(f"archive/{stem}", p, s)
                row.update(strict_graph_valid=strict_valid, archive_total=total,
                           archive_errors=errors)
                actual = (row["result"] or {}).get("our_total")
                row["relative_difference"] = abs(actual - total) / total if actual is not None and total else None
                row["super_native_steps"] = sum(any(x > 128 for x in g) for g in s["granularities"])
                normalized = self.evaluate(f"archive_explicit_raster/{stem}", p,
                                           explicit_raster(p, s))
                normalized_total = (normalized["result"] or {}).get("our_total")
                row["explicit_raster_result"] = normalized
                row["explicit_raster_relative_difference"] = (
                    abs(normalized_total - total) / total if normalized_total is not None and total else None)
                rows.append(row)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/vnrc/evaluate")
    parser.add_argument("--output", type=Path, default=ROOT / "build/vnrc-assessment")
    parser.add_argument("--compare-archive", action="store_true")
    args = parser.parse_args()
    actual_revision = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/vnrc"), "rev-parse", "HEAD"], text=True).strip()
    if actual_revision != REVISION:
        parser.error("VNRC checkout differs from pinned revision")
    a = Assessment(args.binary.resolve(), args.output.resolve())
    report = dict(vnrc_revision=REVISION, platform=platform.platform(), python=sys.version,
                  binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                  assessment_completed_is_not_approval=True)
    for name, method in (("examples", a.examples), ("adversarial", a.adversarial),
                         ("cost_semantics", a.cost_semantics),
                         ("corpus", a.corpus)):
        report[name] = method()
        print(f"Completed {name}: {len(report[name])} cases", flush=True)
    if args.compare_archive:
        report["archive"] = a.archive()
        print(f"Completed archive: {len(report['archive'])} cases", flush=True)
    target = a.output / "summary.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Assessment written to {target}")


if __name__ == "__main__":
    main()
