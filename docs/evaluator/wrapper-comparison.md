# Original Huawei/VNRC CLI versus the validated wrapper

October 30, 2026. **Huawei CSL and VNRC refer to the same source**:
[huawei-csl/mlsys-2026-graph-scheduling-vnrc-submission](https://github.com/huawei-csl/mlsys-2026-graph-scheduling-vnrc-submission).
VNRC is the competition team's name; Huawei CSL hosts its repository. The project
wrapper links that implementation's unchanged core rather than replacing it.

For RL experiments, use the [project wrapper](../../evaluator/README.md).

Repository-layout update, November 16: the current evaluator validates inputs
through `evaluator/problem.py`, and compiler examples now live under
`scripts/hlo2mlsys/`. The 124-workload timings and test counts below describe the
October 30 snapshot, including the 100 synthetic files removed from the current
repository. See the evaluator README for the current 32-test verification scope.

| Property | Original CLI | Project wrapper |
| --- | --- | --- |
| Cost formula | Pinned Huawei/VNRC model | Same model, identified in every result |
| Null/missing traversal | Defaults to snake | Defaults to raster as documented |
| Execution-order / tensor-state validation | Incomplete | Checks availability at every step and final outputs |
| Invalid-schedule probes accepted | 7 of 11 | 0 of 11 in submission mode |
| Arbitrary valid traversal permutations | Silently substitutes raster and can mis-score | Explicitly rejects unsupported patterns |
| Claimed latency labels | Diagnostic only | Verified in submission mode; recomputed for RL rewards |
| Infeasible result output | May contain non-JSON `inf` | Structured invalid result with `total_latency: null` |
| RL access | Process invocation | Reusable Python context calling C++ in process |
| Reward / baseline | No dedicated RL API | Feasible no-fusion baseline and finite bounded reward |

The 13 validation probes comprise 11 expected-invalid schedules, one valid chain,
and one arbitrary valid tile permutation outside our supported subset. The wrapper
accepts the valid chain and explicitly rejects that unsupported permutation.
It does not claim to accept every valid competition-format schedule. Submission
mode is required when testing missing/false reported labels; ordinary RL actions
can omit labels because the evaluator computes the reward itself.

## Comparison on identical inputs and schedules

Both versions evaluated the same 24 official benchmark graphs using identical
deterministic no-fusion schedules. Tile traversal was written explicitly as raster
in both inputs, preventing the CLI default bug from confounding the comparison.
Labels came from the pinned model and were checked by the wrapper.

**All 24 latency totals agreed within the original CLI's two-decimal stdout
rounding (absolute tolerance 0.011).** This establishes parity with that cost
implementation on these schedules, not parity with the competition's private judge.

We measured 9 shuffled rounds after warm-up, alternating approach order, for
216 calls per version on this Mac:

| Execution path | Median wall time | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Original Huawei/VNRC CLI | 4.82 ms | 9.79 ms | 15.40 ms |
| Validated wrapper API | 0.54 ms | 1.33 ms | 1.65 ms |

The ratio of these aggregate medians is approximately **9x**. CLI timing includes
process startup, file/JSON IO, diagnostics, and alternative-tile searches. The API
reuses a problem context but includes schedule validation, explicit permutation
checks, label verification, native subgraph construction/scoring, and result
parsing. This is an interface-level comparison appropriate to repeated RL scoring,
not evidence that the shared mathematical formula became faster. Background
desktop load and caches were uncontrolled, and other schedules can differ.

For ordinary RL actions without explicit permutation lists or predicted latency
labels, a separate complete-API benchmark on 124 workloads measured medians of
0.35 ms for official graphs and 0.44 ms for synthetic graphs. Those are different
fixtures/validation settings, so use the matched table above for direct comparison.

## Remaining limitation

Both versions retain the same large-K/base-cost convention. Full equivalence to
the official competition judge remains unverified. Our wrapper makes that choice
visible and consistent; it does not resolve the external convention disagreement
with the existing ZIP simulator. Record `vnrc-1f19c2f-validated-v1` with training
results and use the same evaluator for every baseline and policy.

## Reproduce

```sh
python3 tools/setup_vnrc.py
python3 -m unittest discover -s tests -v
python3 tools/compare_evaluators.py
```

The matched benchmark writes `build/evaluator-comparison/results.json`. The saved
snapshot `wrapper-comparison.json` also includes the validation-probe comparison.
The current wrapper passed 26 test methods, covering all 11 official worked
examples (snake opt-in where required), all 124 workload baselines, state
validation counterexamples, malformed actions, rewards, and lifecycle handling.
