#!/usr/bin/env python3
"""
hlo2mlsys: generate ML programs -> compile with XLA -> extract the HLO graph
-> write MLSys 2026 contest problem JSON.

Subcommands
  gen      write random JAX model programs (.py)
  compile  lower/compile a program with XLA, dump HLO (pre- and post-optimization)
  convert  turn an HLO text file into a contest problem JSON
  check    validate a contest problem JSON
  all      gen + compile + convert + check, for N seeds

Examples
  python3 pipeline.py all --count 20 --size medium --out out/
  python3 pipeline.py gen --seed 7 --size large --out prog.py
  python3 pipeline.py compile prog.py --out-dir hlo/
  python3 pipeline.py convert hlo/prog.pre.hlo --out prog.json
  python3 pipeline.py check prog.json
"""
import argparse, importlib.util, json, math, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402  (cost simulator, sim.py next to this file)
from collections import defaultdict

# ======================================================================
# 1. PROGRAM GENERATOR
# ======================================================================

SIZES = {
    #          rows (S)           model dim d        ffn dim f              blocks
    "small":  dict(S=[128, 256, 512], d=[128, 256], f=[256, 512],          blocks=(1, 2), max_heads=1),
    "medium": dict(S=[256, 512, 1024], d=[128, 256, 512], f=[512, 1024],   blocks=(2, 4), max_heads=2),
    "large":  dict(S=[512, 1024, 2048], d=[256, 512, 1024], f=[1024, 2048, 4096], blocks=(3, 6), max_heads=3),
}
ACTS = {
    "relu": "jax.nn.relu({})",
    "gelu": "jax.nn.gelu({}, approximate=True)",
    "silu": "jax.nn.silu({})",
    "tanh": "jnp.tanh({})",
}


class ProgramWriter:
    """Builds the source text of a JAX function, one statement per line."""

    def __init__(self, rng, cfg):
        self.rng, self.cfg = rng, cfg
        self.params, self.lines, self.n = [], [], 0
        self.d = rng.choice(cfg["d"])
        self.S = rng.choice(cfg["S"])

    def param(self, prefix, shape):
        name = f"{prefix}{len(self.params)}"
        self.params.append((name, tuple(shape)))
        return name

    def emit(self, expr, prefix="t", comment=None):
        self.n += 1
        v = f"{prefix}{self.n}"
        self.lines.append(f"    {v} = {expr}" + (f"  # {comment}" if comment else ""))
        return v

    def comment(self, text):
        self.lines.append(f"    # --- {text}")

    # ---- building blocks; every block maps x:(S, d) -> (S, d)
    def linear(self, x, din, dout, bias=None):
        w = self.param("w", (din, dout))
        y = self.emit(f"{x} @ {w}")
        if bias if bias is not None else self.rng.random() < 0.5:
            b = self.param("b", (dout,))
            y = self.emit(f"{y} + {b}")
        return y

    def mlp(self, x):
        f = self.rng.choice(self.cfg["f"])
        act = self.rng.choice(list(ACTS))
        self.comment(f"MLP d={self.d} -> {f} -> {self.d}, {act}")
        h = self.emit(ACTS[act].format(self.linear(x, self.d, f)))
        return self.linear(h, f, self.d)

    def glu(self, x):
        f = self.rng.choice(self.cfg["f"])
        self.comment(f"gated MLP (SwiGLU) d={self.d} -> {f} -> {self.d}")
        g = self.linear(x, self.d, f, bias=False)
        u = self.linear(x, self.d, f, bias=False)
        h = self.emit(f"jax.nn.silu({g}) * {u}")
        return self.linear(h, f, self.d, bias=False)

    def attention(self, x):
        dh = 128
        heads = max(1, min(self.cfg["max_heads"], self.d // dh))
        self.comment(f"self-attention, {heads} head(s) of dim {dh}")
        out = None
        for _ in range(heads):
            q = self.linear(x, self.d, dh, bias=False)
            k = self.linear(x, self.d, dh, bias=False)
            v = self.linear(x, self.d, dh, bias=False)
            s = self.emit(f"({q} @ {k}.T) * {1 / math.sqrt(dh):.6f}", comment="scores (S x S)")
            p = self.emit(f"jax.nn.softmax({s}, axis=-1)")
            o = self.emit(f"{p} @ {v}")
            proj = self.linear(o, dh, self.d, bias=False)   # per-head output projection
            out = proj if out is None else self.emit(f"{out} + {proj}")
        return out

    def norm(self, x):
        g = self.param("g", (self.d,))
        if self.rng.random() < 0.5:
            self.comment("RMSNorm")
            return self.emit(f"{x} * jax.lax.rsqrt(jnp.mean({x} * {x}, axis=-1, keepdims=True) + 1e-6) * {g}")
        self.comment("LayerNorm")
        m = self.emit(f"{x} - jnp.mean({x}, axis=-1, keepdims=True)")
        return self.emit(f"{m} * jax.lax.rsqrt(jnp.mean({m} * {m}, axis=-1, keepdims=True) + 1e-5) * {g}")

    def chain(self, x):
        n = self.rng.choice([2, 3])
        self.comment(f"chained matmuls x @ W @ W ... ({n})")
        for _ in range(n):
            x = self.emit(f"{x} @ {self.param('w', (self.d, self.d))}")
        return x

    def build(self, family):
        x = self.param("x", (self.S, self.d))
        lo, hi = self.cfg["blocks"]
        nblocks = self.rng.randint(lo, hi)
        if family == "transformer":
            for i in range(nblocks):
                self.comment(f"layer {i}")
                x = self.emit(f"{x} + {self.attention(self.norm(x))}", comment="residual")
                ffn = self.glu if self.rng.random() < 0.5 else self.mlp
                x = self.emit(f"{x} + {ffn(self.norm(x))}", comment="residual")
        elif family == "mlp":
            for _ in range(nblocks):
                y = self.mlp(x)
                x = self.emit(f"{x} + {y}", comment="residual") if self.rng.random() < 0.7 else y
        else:  # "random": random blocks, random skips and branches
            live, used = [x], set()
            blocks = [self.mlp, self.glu, self.attention, self.norm, self.chain]
            for _ in range(nblocks + 1):
                src = self.rng.choice(live[-3:])
                used.add(src)
                y = self.rng.choice(blocks)(src)
                r = self.rng.random()
                if r < 0.35:
                    y = self.emit(f"{src} + {y}", comment="residual")
                elif r < 0.55 and len(live) > 1:
                    other = self.rng.choice(live[:-1])
                    used.add(other)
                    y = self.emit(f"{y} * {ACTS['tanh'].format(other)}", comment="skip/gating branch")
                live.append(y)
                x = y
            # join every branch that nothing consumed, so no block is dead code
            for v in live[:-1]:
                if v not in used:
                    x = self.emit(f"{x} + {v}", comment="join dangling branch")
        return x

    def source(self, family, seed, size):
        out = self.build(family)
        shapes = ",\n".join(f'    "{n}": {s}' for n, s in self.params)
        args = ", ".join(n for n, _ in self.params)
        return (f'"""Generated by hlo2mlsys: family={family} size={size} seed={seed}"""\n'
                "import jax\nimport jax.numpy as jnp\n\n"
                f"INPUT_SHAPES = {{\n{shapes}\n}}\n\n\n"
                f"def model({args}):\n" + "\n".join(self.lines) + f"\n    return {out}\n")


def generate(seed, size="medium", family=None, min_matmuls=2):
    """Deterministic in (seed, size, family). Retries until the program has >= min_matmuls matmuls."""
    for attempt in range(100):
        rng = random.Random(seed * 1000 + attempt)
        fam = family or rng.choice(["transformer", "mlp", "random", "random"])
        src = ProgramWriter(rng, SIZES[size]).source(fam, seed, size)
        code = "\n".join(l.split("#")[0] for l in src.splitlines())
        if code.count(" @ ") >= min_matmuls:
            return src, fam
    raise RuntimeError("could not generate a program with enough matmuls")


# ======================================================================
# 2. COMPILE WITH XLA
# ======================================================================

def compile_program(path, out_dir):
    """Lower + compile model() with XLA. Writes <name>.pre.hlo and <name>.opt.hlo."""
    import jax, jax.numpy as jnp
    jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "cpu"))
    spec = importlib.util.spec_from_file_location("prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = [jax.ShapeDtypeStruct(s, jnp.float32) for s in mod.INPUT_SHAPES.values()]
    lowered = jax.jit(mod.model).lower(*args)
    base = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0])
    os.makedirs(out_dir, exist_ok=True)
    pre = lowered.compiler_ir("hlo").as_hlo_text()          # what XLA receives
    opt = lowered.compile().as_text()                         # after XLA's passes incl. fusion
    with open(base + ".pre.hlo", "w") as f: f.write(pre)
    with open(base + ".opt.hlo", "w") as f: f.write(opt)
    return base + ".pre.hlo", base + ".opt.hlo", jax.devices()[0].platform


# ======================================================================
# 3. HLO PARSER
# ======================================================================

INSTR = re.compile(r"^(?P<root>ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<type>.+?)\s+"
                   r"(?P<op>[a-z][\w\-]*)\((?P<args>[^)]*)\)(?P<attrs>.*)$")
SHAPE = re.compile(r"(\w+)\[([\d,]*)\]")


def parse_hlo(text):
    comps, entry, cur = {}, None, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("HloModule", "//")):
            continue
        if cur is None:
            if line.endswith("{") and "=" not in line.split("(")[0]:
                is_entry = line.startswith("ENTRY")
                name = line[5:].strip() if is_entry else line
                name = re.split(r"[\s(]", name.lstrip("%"), maxsplit=1)[0]
                cur = comps[name] = []
                if is_entry:
                    entry = name
            continue
        if line == "}":
            cur = None
            continue
        m = INSTR.match(line)
        if not m:
            continue
        t = m["type"]
        dims = None
        if not t.startswith("("):
            sm = SHAPE.match(t)
            dims = [int(x) for x in sm.group(2).split(",") if x] if sm else []
        args = [] if m["op"] in ("constant", "parameter", "iota") else \
            [a.strip().split()[-1].lstrip("%") for a in m["args"].split(",") if a.strip()]
        cur.append(dict(name=m["name"], op=m["op"], dims=dims, args=args, attrs=m["attrs"],
                        root=bool(m["root"]), raw_args=m["args"]))
    if entry is None:
        raise ValueError("no ENTRY computation found")
    return comps, entry


def attr(attrs, key, default=None):
    m = re.search(r"\b" + key + r"=(?:\{([^}]*)\}|%?([\w.\-]+))", attrs)
    if not m:
        return default
    return m.group(1) if m.group(1) is not None else m.group(2)


# ======================================================================
# 4. HLO -> CONTEST GRAPH
# ======================================================================

# Relative cost of one native 128x128 tile for each elementwise HLO opcode.
POINTWISE_COST = {
    "add": 25, "subtract": 25, "multiply": 25, "maximum": 25, "minimum": 25, "negate": 25,
    "abs": 25, "select": 25, "compare": 25, "clamp": 25, "sign": 25, "and": 25, "or": 25,
    "not": 25, "xor": 25, "floor": 25, "ceil": 25, "round-nearest-even": 25,
    "divide": 50, "sqrt": 50, "rsqrt": 50, "remainder": 50, "power": 75,
    "exponential": 100, "exponential-minus-one": 100, "log": 100, "log-plus-one": 100,
    "tanh": 100, "logistic": 100, "erf": 100, "sine": 100, "cosine": 100,
    "reduce": 75, "transpose": 100,
    # Each final Pointwise op costs max(100, sum of its merged HLO ops), matching the
    # 100-500 range of the official benchmarks. MatMul cost is --matmul-cost (default 2000).
}
ALIAS = {"copy", "convert", "bitcast", "reshape", "broadcast", "bitcast-convert", "optimization-barrier"}
NONE_OPS = {"constant", "iota", "rng", "rng-bit-generator", "after-all", "partition-id", "replica-id"}


class Unsupported(Exception):
    pass


class GraphBuilder:
    def __init__(self, matmul_cost):
        self.tensors, self.ops = [], []
        self.matmul_cost = matmul_cost

    def tensor(self, h, w):
        self.tensors.append((h, w))
        return len(self.tensors) - 1

    def op(self, kind, inputs, shape, cost, hlo, fusion, opcode):
        out = self.tensor(*shape)
        ins = list(dict.fromkeys(inputs))          # dedupe, keep order (x*x -> one input)
        self.ops.append(dict(type="MatMul" if kind == "matmul" else "Pointwise", kind=kind,
                             inputs=ins, outputs=[out], cost=cost, hlo=[hlo], fusion=fusion,
                             opcodes=[opcode]))
        return out

    def transpose(self, t, hlo, fusion):
        h, w = self.tensors[t]
        return self.op("transpose", [t], (w, h), POINTWISE_COST["transpose"], hlo + ".T", fusion, "transpose")

    # ---- evaluate one HLO computation symbolically
    def run(self, comps, name, args=None, fusion=None):
        env, root = {}, None
        for ins in comps[name]:
            op, dims, a = ins["op"], ins["dims"], ins["args"]
            if dims is not None and len(dims) > 2:
                raise Unsupported(f"rank-{len(dims)} tensor in {ins['name']} ({op})")
            vals = [env.get(x) for x in a]
            tids = [v for v in vals if isinstance(v, int)]
            if op == "parameter":
                idx = int(ins["raw_args"] or 0)
                if args is not None:
                    v = args[idx]
                else:   # graph input: 2-D tensors become inputs; vectors/scalars (bias, gain) are dropped
                    v = self.tensor(*dims) if dims is not None and len(dims) == 2 else None
            elif op in NONE_OPS:
                v = self.tensor(*dims) if dims is not None and len(dims) == 2 and min(dims) > 1 else None
            elif op in ALIAS:
                v = vals[0] if vals else None
            elif op == "tuple":
                v = vals
            elif op == "get-tuple-element":
                v = vals[0][int(attr(ins["attrs"], "index"))]
            elif op == "fusion" or op == "call":
                callee = attr(ins["attrs"], "calls") or attr(ins["attrs"], "to_apply")
                v = self.run(comps, callee, vals, fusion or (ins["name"] if op == "fusion" else None))
            elif op == "transpose":
                v = self.transpose(tids[0], ins["name"], fusion) if tids else None
            elif op == "dot":
                if attr(ins["attrs"], "lhs_batch_dims") not in (None, ""):
                    raise Unsupported(f"batched dot {ins['name']}")
                lc = int(attr(ins["attrs"], "lhs_contracting_dims", "1") or 1)
                rc = int(attr(ins["attrs"], "rhs_contracting_dims", "0") or 0)
                l, r = vals
                if l is None or r is None:
                    raise Unsupported(f"dot with non-tensor operand {ins['name']}")
                if lc == 0: l = self.transpose(l, a[0], fusion)
                if rc == 1: r = self.transpose(r, a[1], fusion)
                (lh, lw), (rh, rw) = self.tensors[l], self.tensors[r]
                if lw != rh:
                    raise Unsupported(f"dot shape mismatch {ins['name']}: {lh}x{lw} @ {rh}x{rw}")
                v = self.op("matmul", [l, r], (lh, rw), self.matmul_cost, ins["name"], fusion, "dot")
            elif op == "reduce" or op in POINTWISE_COST:
                if not tids:
                    v = None
                else:
                    # reduce results (row statistics) are modeled as full-shape tensors that a
                    # Pointwise op produces; broadcasts of them alias back to that shape.
                    shape = tuple(dims) if dims and len(dims) == 2 and min(dims) > 1 else self.tensors[tids[0]]
                    kind = "reduce" if op == "reduce" else "pointwise"
                    v = self.op(kind, tids if op != "reduce" else tids[:1], shape,
                                POINTWISE_COST[op], ins["name"], fusion, op)
            else:
                raise Unsupported(f"HLO op '{op}' ({ins['name']})")
            env[ins["name"]] = v
            if ins["root"]:
                root = v
        if root is None:
            root = env[comps[name][-1]["name"]]
        return root


def coarsen(g, respect_fusions):
    """Merge chains of pointwise ops (e.g. the ~10 HLO ops of a GELU) into one op:
    producer P is folded into its only consumer C when both are pointwise with the same shape."""
    changed = True
    while changed:
        changed = False
        consumers = defaultdict(list)
        for i, o in enumerate(g.ops):
            if o is None: continue
            for t in o["inputs"]:
                consumers[t].append(i)
        for pi, p in enumerate(g.ops):
            if p is None or p["kind"] in ("matmul", "transpose"): continue
            t = p["outputs"][0]
            cs = consumers.get(t, [])
            if len(set(cs)) != 1: continue
            ci = cs[0]
            c = g.ops[ci]
            if c["kind"] in ("matmul", "transpose"): continue
            if g.tensors[c["outputs"][0]] != g.tensors[t]: continue
            if respect_fusions and c["fusion"] != p["fusion"]: continue
            new_inputs = []
            for x in c["inputs"]:
                new_inputs += p["inputs"] if x == t else [x]
            c["inputs"] = list(dict.fromkeys(new_inputs))
            c["cost"] += p["cost"]
            c["hlo"] = p["hlo"] + c["hlo"]
            c["opcodes"] = p["opcodes"] + c["opcodes"]
            g.ops[pi] = None
            changed = True
            break


def finalize(g):
    """Drop removed ops and unused tensors, renumber everything."""
    ops = [o for o in g.ops if o is not None]
    used = sorted({t for o in ops for t in o["inputs"] + o["outputs"]})
    remap = {t: i for i, t in enumerate(used)}
    for o in ops:
        o["inputs"] = [remap[t] for t in o["inputs"]]
        o["outputs"] = [remap[t] for t in o["outputs"]]
    return [g.tensors[t] for t in used], ops


# (max ops, capacity in native 128x128 tiles, bandwidth) — fitted to the 24 official benchmarks
HW_BANDS = [(10, (1.8, 3.7), (10, 20)), (25, (1.5, 2.8), (12, 15)), (45, (9, 15), (20, 25)),
            (80, (18, 37), (30, 50)), (10**9, (27, 49), (60, 100))]


def pick_hardware(ops, rng, capacity=None, bandwidth=None):
    """Capacity and bandwidth drawn from the ranges the official benchmarks use for graphs of this size."""
    n = len(ops)
    _, (clo, chi), (blo, bhi) = next(b for b in HW_BANDS if n < b[0])
    if capacity is None:
        capacity = max(5000, round(rng.uniform(clo, chi) * 128 * 128 / 5000) * 5000)
    if bandwidth is None:
        bandwidth = rng.randint(blo, bhi)
    return capacity, bandwidth


def convert(hlo_text, matmul_cost=2000, coarse=True, seed=0, capacity=None, bandwidth=None):
    comps, entry = parse_hlo(hlo_text)
    g = GraphBuilder(matmul_cost)
    g.run(comps, entry)
    has_fusions = any(o["fusion"] for o in g.ops)
    if coarse:
        coarsen(g, respect_fusions=has_fusions)
    tensors, ops = finalize(g)
    if not ops:
        raise Unsupported("graph has no ops after conversion")
    cap, bw = pick_hardware(ops, random.Random(seed), capacity, bandwidth)
    problem = {
        "widths": [w for h, w in tensors],
        "heights": [h for h, w in tensors],
        "inputs": [o["inputs"] for o in ops],
        "outputs": [o["outputs"] for o in ops],
        "base_costs": [o["cost"] if o["type"] == "MatMul" else max(100, o["cost"]) for o in ops],
        "op_types": [o["type"] for o in ops],
        "fast_memory_capacity": cap,
        "slow_memory_bandwidth": bw,
        "native_granularity": [128, 128],
    }
    groups = defaultdict(list)
    for i, o in enumerate(ops):
        if o["fusion"]: groups[o["fusion"]].append(i)
    meta = {
        "ops": [dict(index=i, type=o["type"], hlo_instructions=o["hlo"], hlo_opcodes=o["opcodes"],
                     xla_fusion=o["fusion"], is_transpose=o["kind"] == "transpose") for i, o in enumerate(ops)],
        "xla_fusion_groups": {k: v for k, v in groups.items()},
    }
    return problem, meta


def mermaid(problem):
    lines = ["graph LR"]
    for t, (w, h) in enumerate(zip(problem["widths"], problem["heights"])):
        lines.append(f'    Tensor{t}(("Tensor[{t}]<br>{h}x{w}"))')
    for i, (ins, outs, c, ty) in enumerate(zip(problem["inputs"], problem["outputs"],
                                               problem["base_costs"], problem["op_types"])):
        lines.append(f'    Op{i}["Op[{i}] {ty}<br>{c}"]')
        lines += [f"    Tensor{t} --> Op{i}" for t in ins] + [f"    Op{i} --> Tensor{t}" for t in outs]
    return "\n".join(lines) + "\n"


# ======================================================================
# 5. VALIDATOR
# ======================================================================

def check(problem, meta=None):
    """Returns (errors, warnings, stats)."""
    E, W = [], []
    keys = ["widths", "heights", "inputs", "outputs", "base_costs", "op_types",
            "fast_memory_capacity", "slow_memory_bandwidth", "native_granularity"]
    for k in keys:
        if k not in problem: E.append(f"missing key {k}")
    if E: return E, W, {}
    nt, no = len(problem["widths"]), len(problem["op_types"])
    if len(problem["heights"]) != nt: E.append("widths/heights length differ")
    for k in ("inputs", "outputs", "base_costs"):
        if len(problem[k]) != no: E.append(f"{k} has {len(problem[k])} entries, expected {no}")
    producer, consumed = {}, set()
    for i in range(no):
        for t in problem["inputs"][i] + problem["outputs"][i]:
            if not 0 <= t < nt: E.append(f"op {i} references tensor {t} out of range")
        for t in problem["outputs"][i]:
            if t in producer: E.append(f"tensor {t} produced twice (ops {producer[t]}, {i})")
            producer[t] = i
        consumed.update(problem["inputs"][i])
    # DAG + topological order
    for i in range(no):
        for t in problem["inputs"][i]:
            if t in producer and producer[t] >= i:
                E.append(f"op {i} reads tensor {t} produced later by op {producer[t]} (not topological)")
    H, Wd = problem["heights"], problem["widths"]
    transposes = {o["index"] for o in (meta or {}).get("ops", []) if o.get("is_transpose")}
    for i, (ins, outs, ty) in enumerate(zip(problem["inputs"], problem["outputs"], problem["op_types"])):
        o = outs[0]
        if ty == "MatMul":
            if len(ins) != 2: E.append(f"MatMul op {i} has {len(ins)} inputs"); continue
            l, r = ins
            if Wd[l] != H[r] or H[o] != H[l] or Wd[o] != Wd[r]:
                E.append(f"MatMul op {i}: ({H[l]}x{Wd[l]}) @ ({H[r]}x{Wd[r]}) -> ({H[o]}x{Wd[o]})")
        elif ty == "Pointwise":
            for t in ins:
                if (H[t], Wd[t]) != (H[o], Wd[o]):
                    msg = f"Pointwise op {i}: input {H[t]}x{Wd[t]} vs output {H[o]}x{Wd[o]}"
                    (W if i in transposes else E).append(msg + (" (transpose)" if i in transposes else ""))
        else:
            E.append(f"op {i} has unknown type {ty}")
    for k, v in enumerate(H + Wd):
        if v % 128: W.append("a tensor dimension is not a multiple of 128"); break
    stats = dict(ops=no, tensors=nt,
                 matmul=problem["op_types"].count("MatMul"),
                 pointwise=problem["op_types"].count("Pointwise"),
                 graph_inputs=sum(1 for t in range(nt) if t not in producer),
                 graph_outputs=sum(1 for t in range(nt) if t in producer and t not in consumed),
                 edges=sum(len(x) for x in problem["inputs"]) + sum(len(x) for x in problem["outputs"]))
    return E, W, stats


# ======================================================================
# CLI
# ======================================================================

def xla_groups(meta, n_ops):
    """Ops grouped the way XLA fused them, in XLA's execution order. Ops XLA ran
    as standalone kernels become single-op groups."""
    groups, cur, cur_key = [], [], None
    for o in meta["ops"]:
        key = o["xla_fusion"]
        if key is not None and key == cur_key:
            cur.append(o["index"])
        else:
            if cur: groups.append(cur)
            cur, cur_key = [o["index"]], key
    if cur: groups.append(cur)
    return groups


def xla_solution(problem, meta):
    """Contest solution derived from XLA: subgraphs = XLA fusions (execution order);
    granularity + traversal chosen by the simulator; latencies computed by the simulator.
    A fusion that cannot fit fast memory at any granularity is split into single ops."""
    P = sim.Problem(problem)
    groups, split = [], []
    for g in xla_groups(meta, P.n_ops):
        if len(g) > 1 and sim.best_subgraph(P, g)[1] is None:
            split.append(g)
            groups += [[i] for i in g]
        else:
            groups.append(g)
    sol, total = sim.solution_for(P, groups)
    nofusion = sim.solution_for(P, [[i] for i in range(P.n_ops)])[1]
    info = dict(xla_latency=round(total, 4), nofusion_latency=round(nofusion, 4),
                speedup_vs_nofusion=round(nofusion / total, 4), xla_groups=len(groups),
                fused_groups=sum(len(g) > 1 for g in groups), split_fusions=split)
    return sol, info


def write_solution(sol, info, out_json):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f: json.dump(sol, f, indent=2)
    with open(out_json[:-5] + ".info.json", "w") as f: json.dump(info, f, indent=2)


def write_problem(problem, meta, out_json):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f: json.dump(problem, f, indent=2)
    base = out_json[:-5] if out_json.endswith(".json") else out_json
    with open(base + ".meta.json", "w") as f: json.dump(meta, f, indent=2)
    with open(base + ".mmd", "w") as f: f.write(mermaid(problem))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    a = sp.add_parser("gen"); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--size", choices=SIZES, default="medium")
    a.add_argument("--family", choices=["transformer", "mlp", "random"])
    a.add_argument("--out", required=True)

    a = sp.add_parser("compile"); a.add_argument("program"); a.add_argument("--out-dir", default=".")

    for name in ("convert", "all"):
        a = sp.add_parser(name)
        if name == "convert":
            a.add_argument("hlo"); a.add_argument("--out", required=True); a.add_argument("--seed", type=int, default=0)
        else:
            a.add_argument("--count", type=int, default=10); a.add_argument("--start-seed", type=int, default=0)
            a.add_argument("--size", choices=list(SIZES) + ["mixed"], default="mixed")
            a.add_argument("--family", choices=["transformer", "mlp", "random"])
            a.add_argument("--stage", choices=["pre", "opt", "both"], default="opt",
                           help="opt (default): XLA-optimized graph + XLA-derived solution; pre: graph before XLA passes (no solution)")
            a.add_argument("--out", default="out")
        a.add_argument("--matmul-cost", type=int, default=2000, help="base cost of one native MatMul tile")
        a.add_argument("--no-coarsen", action="store_true", help="keep every HLO elementwise op separate")
        a.add_argument("--capacity", type=int); a.add_argument("--bandwidth", type=int)
        a.add_argument("--no-solution", action="store_true", help="skip writing the XLA-derived solution")

    a = sp.add_parser("check"); a.add_argument("json", nargs="+")
    a = sp.add_parser("evaluate", help="recompute a solution's latency with the simulator")
    a.add_argument("problem"); a.add_argument("solution")

    args = ap.parse_args()

    if args.cmd == "gen":
        src, fam = generate(args.seed, args.size, args.family)
        with open(args.out, "w") as f: f.write(src)
        print(f"wrote {args.out} ({fam})")

    elif args.cmd == "compile":
        pre, opt, plat = compile_program(args.program, args.out_dir)
        print(f"XLA ({plat}) -> {pre}, {opt}")

    elif args.cmd == "convert":
        problem, meta = convert(open(args.hlo).read(), args.matmul_cost, not args.no_coarsen,
                                args.seed, args.capacity, args.bandwidth)
        write_problem(problem, meta, args.out)
        E, W, st = check(problem, meta)
        print(f"wrote {args.out}: {st}  errors={len(E)} warnings={len(W)}")
        if meta["xla_fusion_groups"] and not args.no_solution:
            sol, info = xla_solution(problem, meta)
            out = args.out[:-5] + ".solution.json"
            write_solution(sol, info, out)
            print(f"wrote {out}: XLA latency {info['xla_latency']:.1f} "
                  f"(no-fusion {info['nofusion_latency']:.1f}, {info['speedup_vs_nofusion']:.2f}x)")
        elif not meta["xla_fusion_groups"]:
            print("no XLA fusion info in this HLO (pre-optimization?) - no solution written")

    elif args.cmd == "evaluate":
        P = sim.Problem.load(args.problem)
        total, errs = sim.evaluate_solution(P, json.load(open(args.solution)))
        print(f"total latency {total:.4f}" + ("" if not errs else f"  ({len(errs)} problem(s))"))
        for e in errs[:20]: print("  -", e)
        sys.exit(1 if errs else 0)

    elif args.cmd == "check":
        bad = 0
        for p in [p for p in args.json if not p.endswith(".meta.json")]:
            meta_p = p[:-5] + ".meta.json"
            meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
            E, W, st = check(json.load(open(p)), meta)
            bad += bool(E)
            print(f"{p}: {'OK' if not E else 'INVALID'} {st}")
            for e in E[:10]: print("   error:", e)
            for w in sorted(set(W))[:5]: print("   warning:", w)
        sys.exit(1 if bad else 0)

    elif args.cmd == "all":
        stages = ["pre", "opt"] if args.stage == "both" else [args.stage]
        sizes = list(SIZES) if args.size == "mixed" else [args.size]
        for d in ("programs", "hlo", "problems", "solutions"):
            os.makedirs(os.path.join(args.out, d), exist_ok=True)
        summary = []
        for seed in range(args.start_seed, args.start_seed + args.count):
            size = sizes[seed % len(sizes)]
            name = f"prog-{size}-{seed}"
            prog = os.path.join(args.out, "programs", name + ".py")
            src, fam = generate(seed, size, args.family)
            with open(prog, "w") as f: f.write(src)
            pre, opt, plat = compile_program(prog, os.path.join(args.out, "hlo"))
            hw = (args.capacity, args.bandwidth)   # both stages of one program share the same hardware
            for stage, hlo in (("pre", pre), ("opt", opt)):
                if stage not in stages: continue
                out = os.path.join(args.out, "problems", f"{name}.{stage}.json")
                try:
                    problem, meta = convert(open(hlo).read(), args.matmul_cost, not args.no_coarsen, seed, *hw)
                    hw = (problem["fast_memory_capacity"], problem["slow_memory_bandwidth"])
                except Unsupported as e:
                    print(f"{name} [{stage}] skipped: {e}")
                    summary.append(dict(name=name, stage=stage, family=fam, skipped=str(e)))
                    continue
                meta.update(source_program=prog, hlo_file=hlo, xla_stage=stage, xla_platform=plat, family=fam, size=size)
                write_problem(problem, meta, out)
                E, W, st = check(problem, meta)
                row = dict(name=name, stage=stage, family=fam, size=size, valid=not E,
                           capacity=problem["fast_memory_capacity"],
                           bandwidth=problem["slow_memory_bandwidth"], **st)
                sol_txt = ""
                if stage == "opt" and not args.no_solution and not E:
                    sol, info = xla_solution(problem, meta)
                    write_solution(sol, info, os.path.join(args.out, "solutions", f"{name}.opt.solution.json"))
                    row.update({k: v for k, v in info.items() if k != "split_fusions"},
                               split_fusions=len(info["split_fusions"]))
                    sol_txt = f"xla={info['xla_latency']:<12.1f} {info['speedup_vs_nofusion']:.2f}x vs no-fusion"
                summary.append(row)
                print(f"{name} [{stage}] {fam:<11} ops={st['ops']:<4} matmul={st['matmul']:<3} "
                      f"cap={problem['fast_memory_capacity']:<7} bw={problem['slow_memory_bandwidth']:<4} "
                      f"{'OK' if not E else 'INVALID: ' + E[0]}  {sol_txt}", flush=True)
        with open(os.path.join(args.out, "summary.json"), "w") as f: json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
