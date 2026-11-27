import copy
import ctypes
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import weakref

from .problem import FIELDS, validate

REVISION = "1f19c2ffeae394ab7c5f1d92c06795debf667758"
MODEL_ID = "vnrc-1f19c2f-validated-v1"
REWARD_ID = "baseline-over-baseline-plus-latency-v1"
ROOT = Path(__file__).resolve().parents[1]
MAX_GRID = 2**31 - 1


class InvalidSchedule(ValueError):
    def __init__(self, code, message, step=None):
        super().__init__(message)
        self.code, self.step = code, step


@lru_cache(maxsize=None)
def _library(path):
    try:
        lib = ctypes.CDLL(path)
    except OSError as error:
        raise RuntimeError(f"Cannot load evaluator backend; rerun scripts/setup_vnrc.py: {error}") from error
    lib.evaluator_abi_version.restype = ctypes.c_int
    lib.evaluator_revision.restype = ctypes.c_char_p
    if lib.evaluator_abi_version() != 1 or lib.evaluator_revision().decode() != REVISION:
        raise RuntimeError("Evaluator backend version mismatch; rerun scripts/setup_vnrc.py")
    lib.evaluator_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.evaluator_create.restype = ctypes.c_void_p
    lib.evaluator_score.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.evaluator_score.restype = ctypes.c_void_p
    lib.evaluator_destroy.argtypes = [ctypes.c_void_p]
    lib.evaluator_destroy.restype = None
    lib.evaluator_free.argtypes = [ctypes.c_void_p]
    lib.evaluator_free.restype = None
    return lib


def _decode(lib, pointer):
    if not pointer:
        raise MemoryError("Evaluator backend could not allocate its result")
    try:
        try:
            return json.loads(ctypes.string_at(pointer))
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("Evaluator backend returned invalid JSON") from error
    finally:
        lib.evaluator_free(pointer)


def _ids(value, upper, label, step, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise InvalidSchedule("invalid_schema", f"{label} must be a {'nonempty ' if nonempty else ''}list", step)
    if any(type(i) is not int or not 0 <= i < upper for i in value):
        raise InvalidSchedule("invalid_index", f"{label} contains an invalid index", step)
    if len(set(value)) != len(value):
        raise InvalidSchedule("duplicate_index", f"{label} contains duplicates", step)
    return set(value)


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class Evaluator:
    """A reusable problem context using the explicit, pinned VNRC cost convention.

    Default policy allows raster only. None/omitted traversal means raster.
    `allow_snake=True` additionally accepts the two recognized snake patterns.
    This is a project model, not a claim of full official-judge equivalence.
    """

    def __init__(self, problem, *, allow_snake=False, library=None, expected_model_id=MODEL_ID):
        if not isinstance(problem, dict):
            raise ValueError("problem must be a JSON object")
        if type(allow_snake) is not bool:
            raise ValueError("allow_snake must be a boolean")
        if expected_model_id != MODEL_ID:
            raise ValueError(f"Requested model {expected_model_id!r}, but this evaluator implements {MODEL_ID!r}")
        self._problem = copy.deepcopy(problem)
        self._order = validate(self._problem)
        p = self._problem
        if any(len(o) != 1 for o in p["outputs"]):
            raise ValueError("VNRC backend supports exactly one output per operation")
        if p["native_granularity"][0] != p["native_granularity"][1]:
            raise ValueError("VNRC backend requires square native granularity")
        if len(self._order) > 4096 or len(p["widths"]) > 32768:
            raise ValueError("supported graph limit is 4096 operations and 32768 tensors")
        if any(x > MAX_GRID for x in p["widths"] + p["heights"] + p["native_granularity"]):
            raise ValueError("dimensions exceed the native backend's signed-32-bit range")
        if sum(w * h for w, h in zip(p["widths"], p["heights"])) > 2**60:
            raise ValueError("tensor sizes exceed supported safe arithmetic range")
        if any(x > 2**63 - 1 for x in p["base_costs"] +
               [p["fast_memory_capacity"], p["slow_memory_bandwidth"]]):
            raise ValueError("cost/hardware integer exceeds signed-64-bit range")
        self._allow_snake = allow_snake
        encoded_problem = json.dumps({key: p[key] for key in FIELDS}, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()
        self._problem_sha256 = hashlib.sha256(encoded_problem).hexdigest()
        self._producer = {t: i for i, outputs in enumerate(p["outputs"]) for t in outputs}
        consumed = {t for inputs in p["inputs"] for t in inputs}
        self._inputs = set(range(len(p["widths"]))) - self._producer.keys()
        self._outputs = set(self._producer) - consumed
        self._rank = {op: rank for rank, op in enumerate(self._order)}
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        path = Path(library) if library else ROOT / f"build/evaluator/libevaluator_backend{suffix}"
        if not path.is_file():
            raise RuntimeError("Evaluator backend is missing; run python3 scripts/setup_vnrc.py")
        self._lib = _library(str(path.resolve()))
        error = ctypes.c_void_p()
        self._handle = self._lib.evaluator_create(encoded_problem, ctypes.byref(error))
        if not self._handle:
            detail = _decode(self._lib, error.value) if error.value else {}
            raise ValueError(f"Backend rejected problem: {detail.get('error', 'unknown error')}")
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, self._lib.evaluator_destroy, self._handle)
        self._baseline = None

    @property
    def problem(self):
        return copy.deepcopy(self._problem)

    @property
    def allow_snake(self):
        return self._allow_snake

    @property
    def configuration(self):
        """JSON-compatible model identity to record with runs and checkpoints."""
        return dict(model_id=MODEL_ID, vnrc_revision=REVISION,
                    cost_convention="vnrc_base_cost_amortized_over_k_passes",
                    traversal_policy="raster_or_snake" if self.allow_snake else "raster",
                    problem_sha256=self._problem_sha256, reward_id=REWARD_ID,
                    official_equivalence_verified=False)

    def close(self):
        with self._lock:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _check_open(self):
        if not self._finalizer.alive:
            raise RuntimeError("Evaluator is closed")

    def _normalize(self, schedule, submission):
        if not isinstance(schedule, dict):
            raise InvalidSchedule("invalid_schema", "schedule must be a JSON object")
        groups = schedule.get("subgraphs")
        if not isinstance(groups, list) or not groups:
            raise InvalidSchedule("invalid_schema", "subgraphs must be a nonempty list")
        count = len(groups)
        for key in ("granularities", "tensors_to_retain"):
            if not isinstance(schedule.get(key), list) or len(schedule[key]) != count:
                raise InvalidSchedule("invalid_schema", f"{key} must have one entry per subgraph")
        traversals = schedule.get("traversal_orders", [None] * count)
        if not isinstance(traversals, list) or len(traversals) != count:
            raise InvalidSchedule("invalid_schema", "traversal_orders must have one entry per subgraph")
        reported = schedule.get("subgraph_latencies")
        if submission and reported is None:
            raise InvalidSchedule("invalid_schema", "submission requires subgraph_latencies")
        if "subgraph_latencies" in schedule:
            if not isinstance(reported, list) or len(reported) != count or any(
                not _finite_number(v) or v < 0 for v in reported
            ):
                raise InvalidSchedule("invalid_schema", "subgraph_latencies must contain finite nonnegative numbers")
        p = self._problem
        slow, entering, covered, steps = set(self._inputs), set(), set(), []
        for index, group in enumerate(groups):
            ops = _ids(group, len(self._order), "subgraph", index, nonempty=True)
            gran = schedule["granularities"][index]
            if not isinstance(gran, list) or len(gran) != 3 or any(
                type(v) is not int or not 0 < v <= MAX_GRID for v in gran
            ):
                raise InvalidSchedule("invalid_tiling", "granularity must contain three positive integers", index)
            retain = _ids(schedule["tensors_to_retain"][index], len(p["widths"]), "retention", index)
            produced = {t for op in ops for t in p["outputs"][op]}
            consumed = {t for op in ops for t in p["inputs"][op]}
            missing = consumed - produced - slow - entering
            if missing:
                raise InvalidSchedule("unavailable_tensor", f"input tensors are unavailable: {sorted(missing)}", index)
            if not retain <= produced:
                raise InvalidSchedule("invalid_retention", "can retain only tensors produced by this step", index)
            # Connectivity includes producer-consumer edges and shared inputs.
            owners = {}
            neighbors = {op: set() for op in ops}
            for op in ops:
                for t in set(p["inputs"][op] + p["outputs"][op]):
                    if t in owners:
                        other = owners[t]
                        neighbors[op].add(other)
                        neighbors[other].add(op)
                    else:
                        owners[t] = op
            pending, visited = [next(iter(ops))], set()
            while pending:
                op = pending.pop()
                if op not in visited:
                    visited.add(op)
                    pending.extend(neighbors[op] - visited)
            if visited != ops:
                raise InvalidSchedule("unsupported_subgraph", "subgraph must be connected", index)
            boundary_outputs = produced - consumed
            shapes = {(p["widths"][t], p["heights"][t]) for t in boundary_outputs}
            if len(shapes) != 1:
                raise InvalidSchedule("unsupported_subgraph", "boundary outputs must have one common shape", index)
            width, height = next(iter(shapes))
            w, h, k = gran
            if width % w or height % h:
                raise InvalidSchedule("invalid_tiling", "spatial tile must divide the output shape", index)
            nw, nh = width // w, height // h
            if nw * nh > MAX_GRID:
                raise InvalidSchedule("unsupported_grid", "tile grid exceeds supported signed-32-bit range", index)
            if any(p["widths"][p["inputs"][op][0]] // k > MAX_GRID
                   for op in ops if p["op_types"][op] == "MatMul"):
                raise InvalidSchedule("unsupported_grid", "reduction grid exceeds supported range", index)
            order = traversals[index]
            traversal = 0  # SnakeDir::None = explicit raster, including null/omitted.
            if order is not None:
                _ids(order, nw * nh, "traversal", index)
                if len(order) != nw * nh:
                    raise InvalidSchedule("invalid_traversal", "traversal must visit every tile exactly once", index)
                if all(t == i for i, t in enumerate(order)):
                    traversal = 0
                elif self.allow_snake:
                    row_snake = all(order[i] == (i // nw) * nw +
                                    (i % nw if (i // nw) % 2 == 0 else nw - 1 - i % nw)
                                    for i in range(nw * nh))
                    col_snake = all(order[i] == ((i % nh if (i // nh) % 2 == 0
                                                else nh - 1 - i % nh) * nw + i // nh)
                                    for i in range(nw * nh))
                    if row_snake:
                        traversal = 1
                    elif col_snake:
                        traversal = 2
                    else:
                        raise InvalidSchedule("unsupported_traversal", "only raster and row/column snake are supported", index)
                else:
                    raise InvalidSchedule("unsupported_traversal", "this evaluator is configured for raster only", index)
            steps.append(dict(ops=sorted(ops, key=self._rank.get), granularity=gran,
                              retain=sorted(retain), traversal=traversal))
            covered |= ops
            # Only nonretained boundary outputs are written to slow memory.
            # Ephemerals do not become available merely because they were computed.
            slow |= boundary_outputs - retain
            entering = retain
        if covered != set(self._order):
            raise InvalidSchedule("incomplete_coverage", "every operation must be executed at least once")
        if not self._outputs <= slow:
            raise InvalidSchedule("missing_graph_output", "all graph outputs must finish in slow memory")
        return steps, reported

    def evaluate(self, schedule, *, submission=False):
        """Score RL actions; submission=True additionally requires correct latency labels.

        RL actions may omit labels. Supplied finite labels are diagnostics only
        unless submission=True; the reward always uses recomputed costs.
        """
        with self._lock:
            self._check_open()
            metadata = self.configuration
            try:
                steps, reported = self._normalize(schedule, submission)
                result = _decode(self._lib, self._lib.evaluator_score(
                    self._handle, json.dumps(dict(steps=steps), allow_nan=False).encode()))
                if not isinstance(result, dict) or type(result.get("valid")) is not bool:
                    raise RuntimeError("Evaluator backend returned a malformed result")
                if not result["valid"]:
                    if result.get("code") not in ("unsupported_subgraph", "invalid_tiling", "out_of_memory"):
                        raise RuntimeError(f"Evaluator backend failed: {result}")
                    return dict(result, total_latency=None, **metadata)
                latencies, working_sets = result.get("subgraph_latencies"), result.get("working_sets")
                total = result.get("total_latency")
                if (not _finite_number(total) or total <= 0 or not isinstance(latencies, list)
                    or not isinstance(working_sets, list) or len(latencies) != len(steps)
                    or len(working_sets) != len(steps)
                    or any(not _finite_number(v) or v <= 0 for v in latencies)
                    or any(type(v) is not int or not 0 < v <= self._problem["fast_memory_capacity"]
                           for v in working_sets)
                    or not math.isclose(total, sum(latencies), rel_tol=1e-12)):
                    raise RuntimeError("Evaluator backend returned inconsistent costs")
                if reported is not None:
                    matches = all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-3)
                                  for a, b in zip(reported, result["subgraph_latencies"]))
                    result["reported_latencies_match"] = matches
                    if submission and not matches:
                        raise InvalidSchedule("latency_mismatch", "reported latencies do not match recomputed costs")
                return dict(result, **metadata)
            except InvalidSchedule as error:
                return dict(valid=False, total_latency=None, code=error.code, error=str(error),
                            step=error.step, **metadata)

    def baseline(self):
        """Return a deterministic feasible raster, no-fusion/no-retention baseline."""
        with self._lock:
            self._check_open()
            if self._baseline is None:
                p, granularities = self._problem, []
                for op in self._order:
                    out = p["outputs"][op][0]
                    native = p["native_granularity"][0]
                    w, h = math.gcd(native, p["widths"][out]), math.gcd(native, p["heights"][out])
                    mm = p["op_types"][op] == "MatMul"
                    k = math.gcd(native, p["widths"][p["inputs"][op][0]]) if mm else 1
                    def working_set():
                        return w * k + h * k + w * h if mm else (len(set(p["inputs"][op])) + 1) * w * h
                    while working_set() > p["fast_memory_capacity"]:
                        if w >= h and w > 1:
                            w = math.gcd(w, w // 2)
                        elif h > 1:
                            h = math.gcd(h, h // 2)
                        elif k > 1:
                            k = math.gcd(k, k // 2)
                        else:
                            raise RuntimeError("no feasible singleton baseline tile")
                    granularities.append([w, h, k])
                schedule = dict(subgraphs=[[op] for op in self._order], granularities=granularities,
                                tensors_to_retain=[[] for _ in self._order])
                result = self.evaluate(schedule)
                if not result["valid"]:
                    raise RuntimeError(f"could not establish a feasible baseline: {result}")
                schedule["subgraph_latencies"] = result["subgraph_latencies"]
                self._baseline = dict(schedule=schedule, evaluation=result)
            return copy.deepcopy(self._baseline)

    def reward(self, schedule, *, invalid_reward=-1.0):
        """Finite bounded reward: baseline/(baseline+latency); invalid actions < 0.

        A baseline gets 0.5; faster valid schedules approach 1; every valid
        schedule stays positive, so invalid actions cannot beat slow valid ones.
        """
        if not _finite_number(invalid_reward) or invalid_reward >= 0:
            raise ValueError("invalid_reward must be finite and negative")
        with self._lock:
            result = self.evaluate(schedule)
            if not result["valid"]:
                return dict(reward=float(invalid_reward), evaluation=result)
            if self._baseline is None:
                self.baseline()
            baseline = self._baseline["evaluation"]["total_latency"]
            latency = result["total_latency"]
            # Avoid copying the full baseline schedule on every training reward.
            # The ratio form avoids overflow for large finite latencies.
            reward = 1 / (1 + latency / baseline) if latency <= baseline else (baseline / latency) / (1 + baseline / latency)
            return dict(reward=reward, baseline_latency=baseline, evaluation=result)
