# Comparison with competition references and execution time

Measured October 28, 2026, on branch `nam/evaluator`. These measurements describe
the original CLI; see the [October 30 wrapper](../../evaluator/README.md) for the
current validated API and its separate benchmark.

## Comparison with competition results

The official repository's current default revision is
`9eff933d0c90e80d2e407a9e8a76df693438732e`. We fetched its file inventory,
release list, and pinned `PROBLEM.md` directly from GitHub. The official file
matches our local specification byte-for-byte:

```text
SHA-256: 17c447a0dca139099c28bf810efdb0a8f3f8411cfa50c408ff4d2fc92465385f
```

The [official leaderboard](https://github.com/yarongmu-google/MLSys/blob/9eff933d0c90e80d2e407a9e8a76df693438732e/track_a_dashboard.md)
contains normalized points, defined as `best_latency / submitted_latency`,
rounded to two decimals. Its default branch contains benchmark inputs, diagrams,
documentation, and the evaluator header, but no submitted solution corpus paired
with the organizers' raw judged latencies. There are no published GitHub releases
in that repository. This search does not rule out privately shared results or
data hosted elsewhere.

Therefore we can directly test the organizers' **worked examples**, but cannot
claim to have reproduced their final 24-benchmark submission judgments. An exact
comparison needs the same input, the same submitted schedule, its official
accept/reject result, and its official raw latency. A new run of a participant's
solver can choose a different schedule and is not an equivalent reference.

All 11 executable JSON solutions in the
[official specification](https://github.com/yarongmu-google/MLSys/blob/9eff933d0c90e80d2e407a9e8a76df693438732e/PROBLEM.md)
were compared with their stated total latency:

| Example / strategy | Official expected | Unmodified VNRC CLI |
| --- | ---: | ---: |
| 1A: spill | 6553.6 | 6553.6 |
| 1B: fused, native tile | 3276.8 | 3276.8 |
| 1C: fused, smaller tile | 4400 | 4400 |
| 2A: larger tensors, spill | 26214.4 | 26214.4 |
| 2B: larger tensors, fusion | 13107.2 | 13107.2 |
| 3A: diamond, spill | 11468.8 | 11468.8 |
| 3B: recomputation | 6276.8 | 6276.8 |
| 3C: selective residency | 4638.4 | 4638.4 |
| 4A: default raster traversal | **7096** | **6548** |
| 4B: explicit snake traversal | 6548 | 6548 |
| 5B: split-K | 6915.2 | 6915.2 |

Result: **10/11 totals match unchanged**. Example 4A is understated by **7.72%**
because the CLI interprets null as snake instead of the documented raster
default. Supplying `[0,1,2,3]` explicitly produces 7096, so **11/11 match after
making that traversal explicit**. Example 5A has `NA` instead of a schedule and
is excluded; separate assessment probes cover OOM rejection.

These examples do not establish correctness for every fusion pattern or resolve
the K/base-cost convention disagreement documented in `vnrc-assessment.md`.
The source inventory, hashes, and per-example results are preserved in
`official-comparison.json`.

## Execution time

Environment: macOS 26.6.2 arm64, AppleClang 16.0.0, C++20, `-O2 -UNDEBUG`.
VNRC revision: `1f19c2ffeae394ab7c5f1d92c06795debf667758`.

Each schedule received one warm-up and **9 measured CLI calls**, with schedule
order shuffled reproducibly between rounds. There were **145 distinct schedules
and 1,305 measured calls**, plus warm-ups. Calls executed sequentially under
normal desktop conditions. No filesystem-cache flushing, CPU pinning, or
background-load control was used.

### Full unmodified command-line evaluator

These are wall times for `evaluate problem.json solution.json`, including process
startup, JSON parsing, graph construction, stderr/stdout capture, feasibility and
cost calculations, diagnostics, and the CLI's alternative-tile `best_cost()`
searches. Compilation and fixture generation are excluded.

| Workload group | Schedules | Measured calls | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official worked examples | 11 | 99 | 2.07 ms | 2.92 ms | 3.33 ms |
| Official benchmark graphs | 24 | 216 | 5.11 ms | 11.01 ms | 14.63 ms |
| Synthetic graphs | 100 | 900 | 5.07 ms | 9.22 ms | 13.23 ms |
| Accepted compiler-archive schedules | 10 | 90 | 2.79 ms | 3.80 ms | 4.12 ms |

The official/synthetic corpus uses feasible, topologically ordered singleton
schedules from the prior smoke assessment, not competition submissions or
optimized policies. Their zero reported latency labels can trigger additional
diagnostic searches. Only the 10 compiler schedules already accepted by VNRC
are included in that group's timing; this does not imply they are fully valid.

The slowest schedule by median CLI time is `mlsys-2026-21`, with 152 operations:
**12.99 ms median**, **14.63 ms maximum**.

### Prepared core cost calculation

`tools/vnrc_timing_driver.cpp` links the same unmodified static library and
preconstructs the problem, DAG, subgraphs, tile configurations, and retention
sets. It then times only a complete schedule's repeated `compute_cost()` calls
and retention propagation. It mirrors the current CLI's traversal semantics,
including its documented default bug, so totals can be compared directly.

It excludes process startup, JSON parsing, subgraph construction, full schedule
validation, diagnostic printing, and alternative-tile search. It is **not a
replacement production evaluator**. All 145 core totals matched the CLI totals
within 0.011 absolute units (CLI stdout rounds to two decimals).

Each input uses 7 batches, ending at 10 ms or 100,000 iterations, whichever comes
first. Per-call time is batch duration divided by iteration count. The table
summarizes the per-schedule medians, in **microseconds**, not milliseconds:

| Workload group | Median prepared scoring time |
| --- | ---: |
| Official worked examples | 0.037 microseconds |
| Official benchmark graphs | 1.03 microseconds |
| Synthetic graphs | 1.51 microseconds |
| Accepted compiler-archive schedules | 0.395 microseconds |

The official/synthetic workloads reach at most **3.89 microseconds** by median
prepared scoring time. This suggests an in-process adapter could avoid most CLI
overhead, but parsing, construction and strict validation would add work. It is
not a measured RL training throughput result.

The wall times above measure the **time taken to run the evaluator**. The
evaluator's output latency numbers describe the **simulated accelerator's work**
and should not be read as the evaluator's execution time.

## Reproduce

```sh
python3 tools/setup_vnrc.py
python3 tools/assess_vnrc.py
python3 tools/benchmark_vnrc.py --repeats 9
```

To include the optional compiler-archive comparison/timing, run the assessment
with `--compare-archive` using a Python environment containing NumPy before
running the benchmark. The benchmark itself uses Python's standard library.

Raw timing samples and aggregate statistics are in
`build/vnrc-timing/results.json`; the checked-in snapshot is `vnrc-timing.json`.
`official-comparison.json` records the reference provenance and example results.
The upstream clone remains unmodified. These historical CLI measurements do not
include the subsequent project wrapper's validation or timing changes.
