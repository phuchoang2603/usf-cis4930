# Project scheduling evaluator

This is the validated evaluator for controlled RL experiments. It calls the pinned
[Huawei CSL-hosted VNRC implementation](https://github.com/huawei-csl/mlsys-2026-graph-scheduling-vnrc-submission)
**in process**, with a reusable Python problem context. VNRC is the competition
team's name; Huawei CSL is the repository owner. It does not
launch VNRC's diagnostic executable or search for alternative tiles during scoring.

The default traversal policy is **raster only**. Missing or null traversal means
raster. Explicit unsupported orders are rejected, never silently substituted.
Opt in to row/column snake with `allow_snake=True` or `--allow-snake`.

## Setup

From the repository root, with Git, Python 3.10+, CMake, and a C++20 compiler:

```sh
python3 scripts/setup_vnrc.py --backend-only
bash scripts/check.sh
```

Setup clones the pinned VNRC revision and downloads a checksum-verified
nlohmann/json header if needed. `--backend-only` builds just the shared RL
backend; omit the flag to also build the original solver and diagnostic test
programs. Artifacts are stored under `build/`. No pip packages or system package installation
are required for the Python wrapper. macOS arm64 is tested; Linux build support
is provided by CMake but has not been exercised here.

Graph schema and DAG validation live in `evaluator/problem.py`. The synthetic
workload generator uses this evaluator to check each graph's baseline. To compare
the compiler examples under `scripts/hlo2mlsys/`, run
`python3 scripts/assess_vnrc.py --compare-compiler` (requires NumPy).

Upstream VNRC is Apache 2.0; nlohmann/json is MIT licensed. Their original licenses
are kept with the downloaded sources in `external/`. Upstream remains unmodified.

## Use from RL

Create one evaluator per problem and reuse it for schedules from that episode:

```python
import json
from pathlib import Path
from evaluator import Evaluator

problem = json.loads(Path("benchmarks/mlsys-2026-1.json").read_text())
with Evaluator(problem) as evaluator:
    baseline = evaluator.baseline()
    action = baseline["schedule"]  # Replace with the policy's proposed schedule.
    result = evaluator.evaluate(action)
    reward = evaluator.reward(action)
    run_configuration = evaluator.configuration  # Record with the run/checkpoint.
    print(result["total_latency"], reward["reward"])
```

Required schedule fields are `subgraphs`, `granularities`, and `tensors_to_retain`,
each containing one entry per scheduled step. Traversal and latency labels are
optional for RL actions. The policy chooses the schedule; it does not need to
predict its own latency. Supplied latency labels never determine its reward.

`evaluate()` returns finite JSON-compatible data:

```json
{"valid": false, "total_latency": null, "code": "unavailable_tensor",
 "error": "input tensors are unavailable: [1]", "step": 0}
```

Valid results contain `total_latency`, `subgraph_latencies`, and per-step
`working_sets` in tensor elements. Every result also identifies the model and
traversal policy. No `inf` or `NaN` is emitted. Malformed problems and missing
builds raise setup/input exceptions; invalid schedules return an invalid result.
Backend failures or inconsistent native results raise `RuntimeError` and stop
the scoring call; they do not become invalid-action penalties. This prevents a
training loop from learning around an infrastructure failure.
The evaluator owns a native context, supports `close()`/context managers, copies
the input problem, and serializes calls on each context for thread safety.

The deterministic baseline executes each operation separately in topological
order, writes outputs to slow memory, and chooses fitting divisor tiles. It is a
feasible reference, not an optimized no-fusion schedule.

## Reward

For valid schedules:

```text
reward = baseline_latency / (baseline_latency + schedule_latency)
```

The baseline gets 0.5, faster schedules get a value closer to 1, and slower valid
schedules remain positive. Invalid schedules get -1 by default. This bounded
normalization prevents an invalid action from beating a very slow valid action;
the earlier proposed unbounded negative-ratio reward did not have that property.
An alternative invalid penalty must be finite and negative. Initialization,
baseline generation, and problem parsing should remain outside the policy's
inner scoring loop where practical.

## Validate competition-format submissions

```sh
python3 -m evaluator problem.json solution.json --submission
python3 -m evaluator problem.json solution.json --reward
python3 -m evaluator problem.json
```

The final command emits the feasible baseline and its evaluation. `--submission`
requires finite reported latency labels and verifies each against recomputed
cost with relative tolerance 1e-6 and absolute tolerance 1e-3. Ordinary RL scoring
recomputes the cost and reports whether any supplied labels agree, without using
those labels for acceptance or reward. CLI exit code is 0 on valid input, 2 on
invalid problems/schedules, and 1 on backend/setup failure; stdout is structured
JSON. Training code should treat backend failure as an error, not an action penalty.

Validation covers:

- Schema, integer ranges, shapes, DAGs, unique producers, and one output per op.
- Nonempty connected subgraphs, duplicate indices, coverage, and valid tiling.
- Tensor availability **at the time of each step**, including ephemeral tensors,
  retention from the immediately preceding step, and previously written copies.
- Retention of only tensors produced by the step; final graph outputs in slow memory.
- Memory feasibility and finite recomputed costs from VNRC.
- Complete raster permutations, or recognized row/column snake when enabled.

Recomputation in separate steps is allowed. Retaining an internally consumed
(ephemeral) output explicitly materializes it in fast memory for the next step.
Dropping an incoming retained tensor does not invent an uncharged slow-memory
copy. Older materialized copies remain available because these operations are
deterministic and tensors are immutable in this model.

## Cost convention and compatibility

Model ID: **`vnrc-1f19c2f-validated-v1`**.
VNRC revision: `1f19c2ffeae394ab7c5f1d92c06795debf667758`.

This version deliberately retains VNRC's formula: operation base cost is
amortized over the subgraph's reduction passes (`base_cost / num_k_passes`), with
VNRC's spatial scaling, tiling propagation, and IO accounting. All baselines and
policies must use this same model ID. Record it with checkpoints/results.

`evaluator.configuration` contains the model ID, reward version, immutable
traversal policy, upstream revision, and a SHA-256 fingerprint of the problem's
scheduling fields. Every evaluation includes these identifiers. When restoring
a run, pass `expected_model_id=saved_configuration["model_id"]` to `Evaluator`;
an incompatible model fails immediately instead of silently changing the
training objective. `allow_snake` must be a boolean and is read-only after creation.

For a compute-bound singleton MatMul with 128x128 output, base cost 2000 and tile
128x128x64, VNRC computes 2000 for full K=128, 256, or 512. The existing ZIP's
simulator computes 2000, 4000, and 8000 respectively. **That external semantic
disagreement is not resolved by this wrapper.** No authoritative public ruling
was located; the publicly visible unmerged baseline PR is also participant code,
not an official judge. The wrapper pins an explicit project convention instead
of silently converting costs or claiming equivalence to the competition.

Every result includes `official_equivalence_verified: false`. All 11 published
worked examples match in the supported raster/snake mode with null interpreted
as raster, but this is not proof of full judge equivalence for arbitrary graphs.
For publication/comparison, report these as modeled latencies under the pinned
VNRC convention. Do not mix existing ZIP labels with new rewards without
re-evaluating the schedules under this same convention.

Unsupported cases fail explicitly: non-square native granularity, multiple
outputs per operation, disconnected groups, mismatched sink shapes, arbitrary
tile permutations, upstream-invalid tiling, or graphs exceeding 4096 operations,
32768 tensors, signed-32-bit dimensions/tile grids, or safe signed-64-bit arithmetic
bounds. This is a supported subset rather than a complete official evaluator.

## Verification and timing

```sh
bash scripts/check.sh
python3 scripts/benchmark_evaluator.py
```

The tests cover all 11 official example schedules, all 24 official workload baselines,
and feasible baselines for the 12 compatible unpacked compiler graphs. The other
12 compiler graphs are explicitly rejected for incompatible Pointwise shapes.
The suite also covers
the original invalid-schedule counterexamples, reported-label manipulation,
nonfinite inputs, context cleanup, concurrent reuse, backend-fault handling,
model identity checks, and reward behavior. The optional snake mode
is enabled only for fixtures that need it; raster remains the training default.
The benchmark measures the complete reusable Python-to-C++ evaluation path,
including strict validation and native subgraph construction, not just the core
cost routine. It writes raw samples to `build/evaluator-wrapper-results.json`.

October 30 final measurements: complete API median 0.34 ms on official graphs
and 0.44 ms on synthetic graphs. The reward API, including cached baseline
normalization, measured 0.32 ms and 0.43 ms respectively. Each uses nine rounds
over all 124 singleton baseline schedules; setup/baseline generation is excluded.
These October 30 measurements include 100 legacy synthetic fixtures that were
removed by the later repository restructuring; they are historical results.
The current `scripts/benchmark_evaluator.py` runs the 24 official workloads and
records the actual UTC measurement timestamp.
The timing script records both paths separately. A separate earlier matched-schedule comparison measured
0.54 ms for this wrapper versus 4.82 ms for the original CLI, with identical
scores on all 24 official inputs. See the [comparison and methodology](../docs/evaluator/wrapper-comparison.md).
