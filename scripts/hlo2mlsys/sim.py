"""
Cost simulator for the MLSys 2026 scheduling problem (PROBLEM.md).

Implements: subgraph fusion (ephemeral intermediates), [w, h, k] tiling with spatial
padding, split-K with output-stationary accumulators, raster/snake traversal with
intra-subgraph reuse of unchanged slices, the working-set (OOM) check, and the
per-step roofline max(compute, memory). It reproduces every number in the
PROBLEM.md examples.

Not modeled: tensors_to_retain (inter-subgraph residency) - every subgraph output
that is needed later is written to slow memory and read back.
"""
import json, math
from collections import defaultdict

import numpy as np

NATIVE = 128
MAX_TILES = 4096


class Problem:
    def __init__(self, d, name="problem"):
        self.name = name
        self.W, self.H = d["widths"], d["heights"]
        self.inputs, self.outputs = d["inputs"], d["outputs"]
        self.cost, self.types = d["base_costs"], d["op_types"]
        self.cap, self.bw = d["fast_memory_capacity"], d["slow_memory_bandwidth"]
        self.n_ops = len(self.types)
        self.consumers = defaultdict(list)
        for i, ins in enumerate(self.inputs):
            for t in ins:
                self.consumers[t].append(i)
        self._cache = {}

    @staticmethod
    def load(path):
        with open(path) as f:
            return Problem(json.load(f), name=path)

    def size(self, t):
        return self.H[t] * self.W[t]


def _dim(tag, full, h, w, kk):
    return {"i": min(h, full), "j": min(w, full), "s": min(kk, full), "F": full}[tag]


def analyze(P, ops, gran):
    """Static part of a subgraph at granularity [w, h, k]: boundary tensors, slice tags
    and sizes, per-step op costs, working set. Returns dict or None on OOM."""
    ops = sorted(set(ops))
    opset = set(ops)
    w, h, k = gran
    produced = {t for i in ops for t in P.outputs[i]}
    used = {t for i in ops for t in P.inputs[i]}
    b_in = sorted(t for t in used if t not in produced)
    b_out = sorted(t for t in produced
                   if not P.consumers[t] or any(c not in opset for c in P.consumers[t]))
    grid_t = max(b_out, key=P.size)
    acc = {i for i in ops if P.types[i] == "MatMul" and P.outputs[i][0] in b_out
           and not any(c in opset for c in P.consumers[P.outputs[i][0]])}
    K = max((P.W[P.inputs[i][0]] for i in acc), default=None)
    nS = math.ceil(K / k) if K else 1
    kk = min(k, K) if K else NATIVE

    region = {}

    def need(t, tag):
        old = region.get(t)
        region[t] = tag if old is None else tuple(a if a == b else "F" for a, b in zip(old, tag))

    for t in b_out:
        need(t, ("i", "j"))
    for i in reversed(ops):
        r, c = region.get(P.outputs[i][0], ("i", "j"))
        if P.types[i] == "MatMul":
            L, R = P.inputs[i]
            if i in acc:
                need(L, (r, "s")); need(R, ("s", c))
            else:
                need(L, (r, "F")); need(R, ("F", c))
        else:
            for x in P.inputs[i]:
                need(x, (r, c))

    sz = {t: _dim(region[t][0], P.H[t], h, w, kk) * _dim(region[t][1], P.W[t], h, w, kk)
          for t in b_in + b_out}
    ws = sum(sz.values())
    if ws > P.cap:
        return None

    costs = {}
    for i in ops:
        out = P.outputs[i][0]
        f = 1.0
        for tag, full in zip(region[out], (P.H[out], P.W[out])):
            d = _dim(tag, full, h, w, kk)
            f *= d / NATIVE if tag == "s" else math.ceil(d / NATIVE)
        if P.types[i] == "MatMul":
            f *= (kk if i in acc else P.W[P.inputs[i][0]]) / NATIVE
        costs[i] = P.cost[i] * f

    return dict(ops=ops, b_in=b_in, b_out=b_out, acc=acc, nS=nS, region=region, sz=sz,
                costs=costs, ws=ws, nI=math.ceil(P.H[grid_t] / h), nJ=math.ceil(P.W[grid_t] / w))


def tile_order(nI, nJ, order):
    tiles = []
    for a in range(nI):
        cols = range(nJ) if (order == "raster" or a % 2 == 0) else range(nJ - 1, -1, -1)
        tiles += [(a, b) for b in cols]
    return tiles


def latency(P, A, order="raster"):
    """Total latency of an analyzed subgraph for a traversal order. Vectorized over tiles;
    within a tile the k-steps are first / middle (identical) / last."""
    tiles = np.array(tile_order(A["nI"], A["nJ"], order))
    n = len(tiles)
    a, b = tiles[:, 0], tiles[:, 1]
    first = np.zeros(n, bool); first[0] = True
    last = np.zeros(n, bool); last[-1] = True
    ci = first.copy(); ci[1:] = a[1:] != a[:-1]           # tile row changed vs previous tile
    cj = first.copy(); cj[1:] = b[1:] != b[:-1]
    ni = last.copy(); ni[:-1] = a[1:] != a[:-1]            # changes before the next tile
    nj = last.copy(); nj[:-1] = b[1:] != b[:-1]
    nS = A["nS"]

    def flags(tag):
        T = set(tag)
        ds = "s" in T and nS > 1
        enter = first | ("i" in T and ci) | ("j" in T and cj)   # slice differs from previous tile
        leave = last | ("i" in T and ni) | ("j" in T and nj)    # slice differs in the next tile
        return ds, enter, leave

    z = np.zeros(n)
    m0, mM, mL = z.copy(), z.copy(), z.copy()              # memory elements: first/middle/last k-step
    c0, cM, cL = z.copy(), z.copy(), z.copy()              # compute
    for t in A["b_in"]:
        ds, enter, _ = flags(A["region"][t])
        s = A["sz"][t]
        m0 += s * (enter | ds); mM += s * ds; mL += s * ds
    for t in A["b_out"]:
        ds, _, leave = flags(A["region"][t])
        s = A["sz"][t]
        m0 += s * ds; mM += s * ds; mL += s * (leave | ds)
    for i in A["ops"]:
        ds, enter, _ = flags(A["region"][P.outputs[i][0]])
        cst = A["costs"][i]
        if i in A["acc"]:
            c0 += cst; cM += cst; cL += cst
        else:
            c0 += cst * (enter | ds); cM += cst * ds; cL += cst * ds
    bw = P.bw
    if nS == 1:   # one step per tile: loads of the first step, writes of the last
        mem = np.zeros(n)
        for t in A["b_in"]:
            ds, enter, _ = flags(A["region"][t]); mem += A["sz"][t] * enter
        for t in A["b_out"]:
            ds, _, leave = flags(A["region"][t]); mem += A["sz"][t] * leave
        return float(np.maximum(c0, mem / bw).sum())
    tot = np.maximum(c0, m0 / bw).sum() + np.maximum(cL, mL / bw).sum()
    if nS > 2:
        tot += (nS - 2) * np.maximum(cM, mM / bw).sum()
    return float(tot)


def eval_subgraph_at(P, ops, gran, order="raster"):
    A = analyze(P, ops, gran)
    if A is None:
        return math.inf
    if A["nI"] * A["nJ"] > MAX_TILES:
        return math.inf
    return latency(P, A, order)


def candidates(P, ops):
    outs = [t for i in ops for t in P.outputs[i]]
    top = max(max(P.H[t], P.W[t]) for t in outs)
    sp = [s for s in (32, 64, 128, 256, 512, 1024) if s < 2 * top]
    Ks = [P.W[P.inputs[i][0]] for i in ops if P.types[i] == "MatMul"]
    ks = [k for k in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) if k < 2 * max(Ks)] if Ks else [1]
    return [(w, h, k) for w in sp for h in sp for k in ks]


def best_subgraph(P, ops):
    """Best (latency, granularity, order) for one subgraph; cached."""
    key = tuple(sorted(ops))
    if key not in P._cache:
        best = (math.inf, None, None)
        for g in candidates(P, key):
            A = analyze(P, key, g)
            if A is None or A["nI"] * A["nJ"] > MAX_TILES:
                continue
            for order in ("raster", "snake"):
                if order == "snake" and A["nI"] < 2:
                    continue
                lat = latency(P, A, order)
                if lat < best[0]:
                    best = (lat, g, order)
        P._cache[key] = best
    return P._cache[key]


def solution_for(P, groups):
    """Contest solution JSON for an ordered list of op groups (no retention)."""
    sol = {"subgraphs": [], "granularities": [], "tensors_to_retain": [],
           "traversal_orders": [], "subgraph_latencies": []}
    total = 0.0
    for g in groups:
        lat, gran, order = best_subgraph(P, g)
        if gran is None:
            raise ValueError(f"group {g} does not fit in memory at any granularity")
        A = analyze(P, g, gran)
        sol["subgraphs"].append(sorted(g))
        sol["granularities"].append(list(gran))
        sol["tensors_to_retain"].append([])
        sol["traversal_orders"].append(
            None if order == "raster" else [x * A["nJ"] + y for x, y in tile_order(A["nI"], A["nJ"], order)])
        sol["subgraph_latencies"].append(round(lat, 4))
        total += lat
    return sol, total


def evaluate_solution(P, sol):
    """Recompute a solution's latency. Returns (total, errors)."""
    errs = []
    covered = set()
    avail = {t for t in range(len(P.W)) if not any(t in o for o in P.outputs)}   # graph inputs
    total = 0.0
    for n, (g, gran) in enumerate(zip(sol["subgraphs"], sol["granularities"])):
        if sol.get("tensors_to_retain", [[]] * len(sol["subgraphs"]))[n]:
            errs.append(f"subgraph {n}: tensors_to_retain is not supported by this simulator")
        gset = set(g)
        produced = {t for i in g for t in P.outputs[i]}
        for i in g:
            for t in P.inputs[i]:
                if t not in avail and t not in produced:
                    errs.append(f"subgraph {n}: op {i} reads tensor {t} before it is produced")
        A = analyze(P, g, gran)
        if A is None:
            errs.append(f"subgraph {n}: OOM at granularity {gran}"); continue
        order = sol.get("traversal_orders", [None] * len(g))[n]
        raster = [x * A["nJ"] + y for x, y in tile_order(A["nI"], A["nJ"], "raster")]
        snake = [x * A["nJ"] + y for x, y in tile_order(A["nI"], A["nJ"], "snake")]
        if order is None or order == raster:
            lat = latency(P, A, "raster")
        elif order == snake:
            lat = latency(P, A, "snake")
        else:
            errs.append(f"subgraph {n}: only raster or snake traversal orders are supported"); continue
        total += lat
        rep = sol["subgraph_latencies"][n]
        if abs(rep - lat) > 1e-3 * max(1.0, lat):
            errs.append(f"subgraph {n}: reported latency {rep} != recomputed {lat:.4f}")
        covered |= gset
        avail |= set(A["b_out"])
    missing = set(range(P.n_ops)) - covered
    if missing:
        errs.append(f"ops never executed: {sorted(missing)[:10]}")
    return total, errs
