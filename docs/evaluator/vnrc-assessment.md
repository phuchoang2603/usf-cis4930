# VNRC evaluator assessment

Date: October 28, 2026. Branch: `nam/evaluator`.

Follow-up: [comparison with official references and execution-time measurements](vnrc-validation-and-timing.md).

October 30 update: the [validated project wrapper](../../evaluator/README.md) is
now implemented. The findings below describe the original, unmodified VNRC CLI.

**Verdict: useful source for a cost-model implementation, but the unmodified CLI
is not ready to serve as the project's authoritative validator or RL reward.**
Its core model passes its own example and regression tests. Independent CLI
checks reveal missing validation, incorrect handling of the documented default
traversal, and cost conventions that disagree with our existing simulator.

## Source and build

- Source: [VNRC's second-place Track A submission](https://github.com/huawei-csl/mlsys-2026-graph-scheduling-vnrc-submission).
- Pinned revision: `1f19c2ffeae394ab7c5f1d92c06795debf667758`.
- License: Apache 2.0. The original source and license remain in `external/vnrc/`.
- JSON dependency: nlohmann/json `v3.11.3`, with its MIT license, in `external/deps/`.
  Header SHA-256: `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6`.
- Tested locally on macOS arm64 with AppleClang 16.0.0 and C++20.
  Upstream documents Linux/GCC as its supported environment; Linux was not tested here.
- Built with `-O2 -UNDEBUG` so assertions in the upstream tests actually execute.
  The upstream default Release build disables them with `-DNDEBUG`.
- No upstream source modifications or system package installs were needed.

The clone, downloaded header, binaries, and raw logs are ignored by the parent
repository. The scripts pin their provenance and recreate them; no nested Git
repository is accidentally committed.

## Results

| Check | Result | What this establishes |
| --- | --- | --- |
| Upstream CTest suite | All 48 test programs passed across the initial run and one timeout rerun | Internal consistency across its cost-model, scheduling, and search tests |
| Upstream `problem_examples_test` | 65 checks passed, 0 failed | Core API reproduces its hand-written examples |
| Actual CLI on all 11 JSON solutions in our `PROBLEM.md` | 10 latency totals matched unchanged | One CLI default differs from the core test's explicit raster configuration |
| Dataset smoke test | 24 official + 100 synthetic workloads accepted | Topologically ordered singleton schedules with fitting tiles are evaluable |
| CLI acceptance probes | 7 of 11 expected-invalid schedules accepted; 2 valid controls accepted | CLI success is insufficient to establish schedule validity under our checked-in specification |
| Valid nonstandard traversal | Accepted, but scored incorrectly | Only recognized traversal patterns are modeled faithfully |
| Existing compiler ZIP | 10/24 schedules accepted; 7 agree with its simulator within 0.01% | Existing graph/schedule generation needs reconciliation too |

The initial CTest run used a 45-second per-test timeout. `constructed_optima_test`
hit that limit; it passed in 47.51 seconds when rerun with a 120-second limit.
This is a solver-heavy test, not evidence of a cost-model failure.

The dataset smoke test checks feasibility and coverage only. It deliberately
supplies zero latency labels, which this CLI ignores for acceptance. It does not
establish accurate latency across the dataset, solver quality, or policy transfer.

## Reproduced issues

### Default traversal changes the documented score

Our `PROBLEM.md` says omitted/null traversal means raster order. Example 4A's
expected total is **7096**. VNRC's CLI treats null as row-major snake and returns
**6548**. Replacing null with the explicit raster permutation `[0,1,2,3]` produces
the correct **7096**. Example 4B's explicit snake permutation also matches **6548**.
Consequently, all 11 documented solution totals match when the affected default
is made explicit.

An arbitrary valid permutation `[0,3,1,2]` should cost **7644** for that same
four-tile example: three visits cost 2048 each, and one visit reuses RHS and costs
1500. The CLI silently maps it to raster and reports **7096**. Duplicate traversal
indices `[0,0,0,0]` also receive a success exit code and the raster score.

### CLI validation is incomplete

Reproducible probes show success for:

- A consumer step scheduled before its producer.
- A consumer of an intermediate that became ephemeral in an earlier fused group
  and was neither materialized nor retained.
- Retaining a graph input instead of an output, under the checked-in specification's
  output-only retention rule. Capacity is deliberately large enough to avoid an
  unrelated OOM rejection.
- Duplicate operator IDs within one subgraph.
- Duplicate traversal indices.
- False reported latency values.
- A missing required `subgraph_latencies` field.

The CLI does reject incomplete coverage, mismatched principal array lengths,
out-of-memory schedules, and zero tile width. For infeasible costs, however, its
stdout includes `inf`, which is not valid JSON. Callers must check the exit code
and reject nonfinite values rather than assume stdout is always parseable JSON.

These acceptance expectations follow our checked-in `PROBLEM.md`. Upstream code
also references competition issue rulings, so rule differences must be reconciled
before declaring exact compatibility with the organizers' evaluator.

The upstream `Solution::validate()` contains additional ordering and ephemeral
checks that the standalone CLI does not call. It is worth reusing, but its
constructor filters retention requests, and some validation checks consider
availability across the whole schedule. A wrapper still needs explicit schema,
range, and state validation; calling that method alone is not a complete fix.

### The two cost models disagree when K exceeds native depth

Three compute-bound singleton MatMul probes use output 128x128, native depth 128,
tile `[128,128,64]`, base cost 2000, and high bandwidth to isolate compute:

| Full reduction K | VNRC CLI | Existing ZIP's native-tile convention |
| --- | ---: | ---: |
| 128 | 2000 | 2000 |
| 256 | 2000 | 4000 |
| 512 | 2000 | 8000 |

VNRC scales compute by `base_cost / num_k_passes`. Our ZIP simulator scales a
MatMul step by `base_cost * k / 128`. These agree when K=128 and diverge otherwise.
The examples alone do not resolve this disagreement. The ZIP column describes
its implemented convention, not independently verified official ground truth.
This convention must be settled before selecting either implementation as the
canonical reward model.

### Existing compiler schedules are not interchangeable

Of the ZIP's 24 graphs, 12 fail our existing strict graph validator because a
Pointwise operation changes shape. All 12 associated schedules were rejected by
VNRC too, but that does not isolate the reason for rejection. Of the 12 graphs
that do pass strict validation:

- 2 have schedules containing tile dimensions greater than native size, which
  VNRC rejects. Its source attributes this restriction to competition rulings.
- 7 schedules are accepted and their totals match the ZIP simulator within 0.01%.
- 3 schedules are accepted but totals differ by approximately 2.49%, 4.55%, and
  4.70%. Making null raster traversals explicit does not remove these differences.
  Their MatMul compute terms expose the cost convention disagreement above.

The ZIP simulator accepts all its own 24 schedules. That is internal consistency,
not external validation of those graphs or latency labels.

## Reproduce

From the repository root, with Git, CMake, a C++20 compiler, and Python available:

```sh
python3 tools/setup_vnrc.py
ctest --test-dir build/vnrc -j2 --output-on-failure --timeout 120
python3 tools/assess_vnrc.py
```

The optional comparison executes `sim.py` directly from the existing ZIP and
requires NumPy. Use a Python environment containing NumPy (tested with 2.3.5):

```sh
python3 tools/assess_vnrc.py --compare-archive
```

The assessment writes inputs, schedules, stdout/stderr, and `summary.json` under
`build/vnrc-assessment/`. Exit 0 means the assessment completed, not that the
candidate evaluator passed every acceptance criterion. Inspect the recorded
`matches`, `matches_cost`, and cost-comparison fields. The checked-in
`vnrc-results.json` preserves the assessed results; raw logs stay in the local
build directory. The setup script requires a clean checkout at the pinned
revision and verifies the downloaded header checksum.

## Recommended adoption path

1. Resolve the K/base-cost convention against the competition's clarifications
   or another independently verified implementation.
2. Keep VNRC as a pinned external dependency and reuse its core cost API behind
   a strict project-owned adapter.
3. Validate tensor availability as execution progresses, including ephemeral
   intermediates and final graph outputs; validate all schema fields and indices.
4. Make null mean raster, and either implement arbitrary permutations or reject
   unsupported permutations explicitly. Never substitute another traversal silently.
5. Return finite machine-readable results and recomputed latency; handle invalid
   schedules predictably before exposing the evaluator as an RL reward.
6. Recheck compiler-generated graphs and schedules under the agreed semantics
   before treating them as training targets or baseline measurements.

The original October 28 assessment included no adapter or upstream patch.
The October 30 project wrapper addresses validation and traversal handling while
preserving the pinned VNRC cost convention; see its README for current scope.
