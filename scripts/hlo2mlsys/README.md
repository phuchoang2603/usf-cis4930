# hlo2mlsys

Generate ML model programs, compile them with a real compiler (XLA, through JAX), and turn the compiler's result into **input/output pairs** for the MLSys 2026 scheduling contest ([PROBLEM.md](https://github.com/yarongmu-google/MLSys/blob/main/PROBLEM.md)):

- **input** `problems/X.opt.json`: the graph XLA produced (contest problem format)
- **output** `solutions/X.opt.solution.json`: a schedule built from XLA's own decisions (contest solution format)

```
random JAX program ──jax.jit──▶ XLA optimized HLO ──convert──▶ problem.json   (input)
                                       │
                                       └─ XLA fusions + execution order ──sim.py──▶ solution.json (output)
```

## Setup

```bash
pip install "jax[cpu]"          # brings XLA; numpy comes with it
```

Python 3.10+. A GPU or TPU is optional. The `opt` graph is whatever XLA produces for the backend JAX finds (`JAX_PLATFORMS=cpu|cuda|tpu`).

## Quick start

```bash
# 20 programs of mixed sizes, both XLA stages -> out/problems/*.json
python3 pipeline.py all --count 20 --out out

# one step at a time
python3 pipeline.py gen --seed 7 --size large --family transformer --out prog.py
python3 pipeline.py compile prog.py --out-dir hlo          # hlo/prog.pre.hlo, hlo/prog.opt.hlo
python3 pipeline.py convert hlo/prog.opt.hlo --out prog.json
python3 pipeline.py check prog.json
```

`convert` also works on any HLO text you dump yourself, for example from your own JAX/TensorFlow model with `XLA_FLAGS=--xla_dump_to=dir`. It works as long as the graph is 2-D matmuls plus elementwise and reduce ops (see Limits below).

## What each step does

### 1. `gen`: write a program

The output is a readable `.py` file with `INPUT_SHAPES` and a `model(...)` function. Every dimension is a multiple of 128.

| Family | Contents |
|---|---|
| `transformer` | Layers of RMSNorm/LayerNorm → multi-head self-attention → residual → MLP or SwiGLU → residual |
| `mlp` | Stacked MLPs (relu/gelu/silu/tanh, optional bias) with random residuals |
| `random` | Random mix of MLP, gated MLP, attention, norm and chained-matmul blocks. Adds random residuals, gating skip connections and branches; every branch is joined into the output, so nothing is dead code. |

`--size small|medium|large` sets rows (128–2048), model width (128–1024), FFN width (256–4096) and depth. Generation is deterministic in `(seed, size, family)`.

### 2. `compile`: run XLA

`jax.jit(model).lower(...)` produces the graph XLA receives (`*.pre.hlo`). `.compile()` runs XLA's full optimization pipeline and returns the optimized module (`*.opt.hlo`). That pipeline includes algebraic simplification, CSE, layout assignment, transpose folding and operator fusion.

### 3. `convert`: HLO → contest graph

| HLO | Contest |
|---|---|
| 2-D `parameter` | graph input tensor (`height` = rows, `width` = cols) |
| vector/scalar `parameter`, `constant` (bias, norm gain, scale) | dropped; it folds into the op that uses it |
| `dot` | `MatMul` (left = LHS). A transposed operand gets an explicit transpose op first. |
| elementwise ops (`add`, `multiply`, `exponential`, `tanh`, `divide`, `rsqrt`, …) | `Pointwise` |
| `reduce` (softmax/norm statistics) | `Pointwise` whose output keeps the full input shape (row stats treated as broadcast) |
| `transpose` | `Pointwise` with swapped output shape (reported as a warning by `check`) |
| `reshape`, `broadcast`, `copy`, `convert`, `bitcast` | no op; the value passes through |
| `fusion` (opt stage) | inlined; each op records the XLA fusion it came from |

**Coarsening** (on by default; turn off with `--no-coarsen`) merges a chain of pointwise ops when each one feeds exactly one pointwise consumer of the same shape. For example, the ~10 HLO ops of a GELU become one op, which is the granularity the official benchmarks use. In the `opt` stage, merging never crosses an XLA fusion boundary, so the graph keeps XLA's decisions.

**Costs and hardware** are not something XLA provides. They are fitted to the 24 official benchmarks:
- MatMul base cost = `--matmul-cost` (default 2000, one native 128×128×128 tile).
- Pointwise base cost = sum of a per-opcode table over the merged HLO ops, with a minimum of 100.
- Capacity and bandwidth are drawn from the ranges the official benchmarks use for graphs with that many ops. They run from about 2–4 native tiles at bandwidth 10–20 for tiny graphs, up to 27–49 tiles at bandwidth 60–100 for 100+ ops.
- `--capacity` and `--bandwidth` override them. Both stages of one program get the same hardware, so their schedules can be compared.

### 4. `check`

`check` verifies the contest format:
- every key is present and the list lengths agree;
- every tensor id is in range and produced at most once;
- ops are in topological order;
- MatMul shapes satisfy `L.width == R.height`;
- Pointwise inputs have the same shape as the output.

It exits non-zero if any file is invalid.

## The solution (output) side

XLA decides **which ops run together** (its fusions) and **in what order** (the optimized module is scheduled). Those become `subgraphs` in that order. XLA has no notion of this contest's tile sizes or latency model, so for each XLA group `sim.py` picks the granularity `[w, h, k]` and traversal (raster or snake) with the lowest latency, and computes `subgraph_latencies`. `tensors_to_retain` is always empty: the simulator does not model retention.

- `sim.py` reproduces every number in the PROBLEM.md examples (fusion, padding, split-K, snake reuse, OOM).
- If an XLA fusion can't fit fast memory at any granularity, it is split into single ops. Such splits are recorded in the `.info.json` file.
- `X.opt.solution.info.json` holds the XLA latency, the no-fusion latency and the speedup.
- `python3 pipeline.py evaluate problem.json solution.json` recomputes any solution's latency and checks coverage, order and memory.

**What to expect.** On the CPU backend, XLA leaves most matmuls as separate kernels, so its schedule is close to the no-fusion baseline (about 0.95-1.03x). It is a real compiler baseline, not an optimal answer. XLA's GPU and TPU backends fuse more aggressively. To get their decisions, run on a GPU/TPU machine (for example a Colab TPU runtime) with `JAX_PLATFORMS=tpu` or `cuda`; that path is untested here and may hit HLO ops the converter doesn't support yet.

## Output files

For each problem `X.json`:

- `X.json`: the contest problem, same schema as `benchmarks/mlsys-2026-*.json`.
- `X.meta.json`: for every op, the HLO instructions and opcodes it came from and its XLA fusion. It also has `xla_fusion_groups`, the op indices XLA fused together. You can use those as a baseline schedule ("what would XLA's own fusion decisions cost under this model?").
- `X.mmd`: Mermaid diagram in the same style as `benchmarks/mermaid/`.

`all` also writes `programs/`, `hlo/`, `solutions/` and `summary.json`. The summary has op/tensor/edge counts, capacity, bandwidth, XLA latency and no-fusion latency per problem. `all` defaults to `--stage opt` (problem + solution); `--stage both` also writes the pre-optimization graph, which has no solution because XLA hasn't made any decisions yet.

## Pre vs opt stage

- **pre**: framework-level graph, close to how the official benchmarks look.
- **opt**: what XLA turned it into. Usually it has the same MatMuls and more, smaller Pointwise ops, because XLA splits norms and softmax across fusions. XLA can also rewrite attention to compute Kᵀ directly through transposes. These are real compiler decisions, which is the point of this stage.

## Limits

- Only 2-D tensors. Programs with batched/3-D dots, `concatenate`, `slice`, `gather`, convolutions or control flow raise `Unsupported`; `all` skips them and records why in `summary.json`. The generator never emits these.
- Transposes are modeled as Pointwise ops whose output shape is swapped. Strictly, PROBLEM.md's Pointwise assumes equal shapes, so `check` lists these as warnings. If your solver can't handle them, filter on `is_transpose` in the meta file.
- Base costs, capacity and bandwidth are modeling choices, not measurements. Only the graph structure and shapes come from the compiler.
