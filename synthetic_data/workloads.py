import hashlib
import json
import math
import random
from collections import deque
from dataclasses import dataclass


FAMILIES = ("chain", "branch", "diamond", "matmul", "mixed")
MIN_NODES = {"chain": 1, "branch": 3, "diamond": 4, "matmul": 1, "mixed": 5}
FIELDS = (
    "widths", "heights", "inputs", "outputs", "base_costs", "op_types",
    "fast_memory_capacity", "slow_memory_bandwidth", "native_granularity",
)


def _positive(value):
    return type(value) is int and value > 0


def validate(problem):
    if not isinstance(problem, dict) or any(k not in problem for k in FIELDS):
        raise ValueError("problem must contain every competition input field")
    for key in FIELDS[:6] + ("native_granularity",):
        if not isinstance(problem[key], list):
            raise ValueError(f"{key} must be a list")
    widths, heights = problem["widths"], problem["heights"]
    if not widths or len(widths) != len(heights):
        raise ValueError("widths and heights must have equal, nonzero length")
    if not all(_positive(x) for x in widths + heights):
        raise ValueError("tensor dimensions must be positive integers")
    count = len(problem["op_types"])
    if not count or any(len(problem[k]) != count for k in ("inputs", "outputs", "base_costs")):
        raise ValueError("operation arrays must have equal, nonzero length")
    if not all(_positive(c) for c in problem["base_costs"]):
        raise ValueError("base costs must be positive integers")
    if not all(_positive(problem[k]) for k in ("fast_memory_capacity", "slow_memory_bandwidth")):
        raise ValueError("memory capacity and bandwidth must be positive integers")
    native = problem["native_granularity"]
    if len(native) != 2 or not all(_positive(x) for x in native):
        raise ValueError("native_granularity must contain two positive integers")

    producer = {}
    shapes = list(zip(heights, widths))
    for op, (kind, inputs, outputs) in enumerate(zip(
        problem["op_types"], problem["inputs"], problem["outputs"]
    )):
        if kind not in ("Pointwise", "MatMul"):
            raise ValueError(f"operation {op}: unsupported operation type")
        if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or not outputs:
            raise ValueError(f"operation {op}: inputs and outputs must be nonempty lists")
        if any(type(t) is not int or not 0 <= t < len(widths) for t in inputs + outputs):
            raise ValueError(f"operation {op}: invalid tensor index")
        if len(set(outputs)) != len(outputs):
            raise ValueError(f"operation {op}: repeated output tensor")
        if set(inputs) & set(outputs):
            raise ValueError(f"operation {op}: in-place tensors are unsupported")
        for tensor in outputs:
            if tensor in producer:
                raise ValueError(f"tensor {tensor} has multiple producers")
            producer[tensor] = op
        if kind == "MatMul":
            if len(inputs) != 2 or len(outputs) != 1:
                raise ValueError(f"operation {op}: MatMul requires two inputs and one output")
            lhs, rhs, out = (shapes[t] for t in inputs + outputs)
            if lhs[1] != rhs[0] or out != (lhs[0], rhs[1]):
                raise ValueError(f"operation {op}: incompatible MatMul shapes")
        elif len({shapes[t] for t in inputs + outputs}) != 1:
            raise ValueError(f"operation {op}: Pointwise shapes must match without broadcasting")
        if len(set(inputs + outputs)) > problem["fast_memory_capacity"]:
            raise ValueError(f"operation {op}: even a 1x1x1 working set cannot fit")
    parents = [{producer[t] for t in inputs if t in producer} for inputs in problem["inputs"]]
    children = [set() for _ in range(count)]
    for op, deps in enumerate(parents):
        for parent in deps:
            children[parent].add(op)
    indegree = [len(deps) for deps in parents]
    ready = deque(i for i, degree in enumerate(indegree) if degree == 0)
    order = []
    while ready:
        op = ready.popleft()
        order.append(op)
        for child in sorted(children[op]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != count:
        raise ValueError("operation dependencies contain a cycle")
    return order


@dataclass(frozen=True)
class GenerationConfig:
    dimensions: tuple = (128, 256, 512, 1024, 2048)
    memory_capacities: tuple = (25000, 50000, 150000, 300000, 600000)
    bandwidths: tuple = (10, 20, 40, 80, 100)
    native_sizes: tuple = (128,)
    pointwise_costs: tuple = (100, 200, 500, 1000)
    matmul_costs: tuple = (500, 1000, 2000, 5000, 10000)
    matmul_probability: float = 0.45

    def __post_init__(self):
        for name in ("dimensions", "memory_capacities", "bandwidths", "native_sizes",
                     "pointwise_costs", "matmul_costs"):
            values = getattr(self, name)
            if not values or not all(_positive(x) for x in values):
                raise ValueError(f"{name} must contain positive integers")
        if min(self.memory_capacities) < 3:
            raise ValueError("memory capacities must fit at least three elements")
        if not 0 <= self.matmul_probability <= 1:
            raise ValueError("matmul_probability must be between zero and one")


class _Builder:
    def __init__(self, rng, config):
        self.rng = rng
        self.config = config
        self.problem = {key: [] for key in FIELDS[:6]}

    def tensor(self, height, width):
        tensor = len(self.problem["widths"])
        self.problem["widths"].append(width)
        self.problem["heights"].append(height)
        return tensor

    def shape(self, tensor):
        return self.problem["heights"][tensor], self.problem["widths"][tensor]

    def op(self, inputs, matmul=False):
        height, width = self.shape(inputs[0])
        if matmul:
            out_width = self.rng.choice(self.config.dimensions)
            inputs = [inputs[0], self.tensor(width, out_width)]
            width = out_width
        out = self.tensor(height, width)
        self.problem["inputs"].append(inputs)
        self.problem["outputs"].append([out])
        self.problem["op_types"].append("MatMul" if matmul else "Pointwise")
        costs = self.config.matmul_costs if matmul else self.config.pointwise_costs
        self.problem["base_costs"].append(self.rng.choice(costs))
        return out


def generate(family, nodes, seed=0, config=None):
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    if type(nodes) is not int or nodes < MIN_NODES[family]:
        raise ValueError(f"{family} requires at least {MIN_NODES[family]} operations")
    config = config or GenerationConfig()
    rng = random.Random(seed)
    builder = _Builder(rng, config)
    current = builder.tensor(rng.choice(config.dimensions), rng.choice(config.dimensions))
    if family == "branch":
        root = builder.op([current])
        arms = [builder.op([root]), builder.op([root])]
        while len(builder.problem["op_types"]) < nodes:
            arm = rng.randrange(2)
            arms[arm] = builder.op([arms[arm]])
    elif family in ("diamond", "mixed"):
        current = builder.op([current])
        while len(builder.problem["op_types"]) + 3 <= nodes - (family == "mixed"):
            left = builder.op([current])
            right = builder.op([current])
            current = builder.op([left, right])
            if family == "mixed":
                break
        if family == "mixed":
            current = builder.op([current], matmul=True)
        pool = [out[0] for out in builder.problem["outputs"]]
        while len(builder.problem["op_types"]) < nodes:
            source = rng.choice(pool) if family == "mixed" else current
            compatible = [t for t in pool if t != source and builder.shape(t) == builder.shape(source)]
            inputs = [source]
            if compatible and rng.random() < 0.4:
                inputs.append(rng.choice(compatible))
            current = builder.op(inputs, matmul=family == "mixed" and len(inputs) == 1
                                 and rng.random() < config.matmul_probability)
            pool.append(current)
    else:
        for op in range(nodes):
            matmul = family == "matmul" and (op == 0 or rng.random() < config.matmul_probability)
            current = builder.op([current], matmul=matmul)
    problem = builder.problem
    native = rng.choice(config.native_sizes)
    problem.update(
        fast_memory_capacity=rng.choice(config.memory_capacities),
        slow_memory_bandwidth=rng.choice(config.bandwidths),
        native_granularity=[native, native],
    )
    validate(problem)
    return problem


def augment(problem, seed=0, dimension_scale=1, cost_scale=1.0,
            memory_scale=1.0, bandwidth_scale=1.0):
    validate(problem)
    if not _positive(dimension_scale):
        raise ValueError("dimension_scale must be a positive integer to preserve shape equalities")
    if any(not math.isfinite(x) or x <= 0 for x in (cost_scale, memory_scale, bandwidth_scale)):
        raise ValueError("cost, memory, and bandwidth scales must be finite and positive")
    result = json.loads(json.dumps(problem))
    rng = random.Random(seed)
    for key in ("widths", "heights"):
        result[key] = [x * dimension_scale for x in result[key]]
    result["base_costs"] = [max(1, round(x * cost_scale * rng.uniform(0.8, 1.2)))
                            for x in result["base_costs"]]
    result["fast_memory_capacity"] = max(3, round(problem["fast_memory_capacity"] * memory_scale))
    result["slow_memory_bandwidth"] = max(1, round(problem["slow_memory_bandwidth"] * bandwidth_scale))
    validate(result)
    return result


def topology_id(problem):
    payload = {key: problem[key] for key in ("inputs", "outputs", "op_types")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def statistics(problem):
    order = validate(problem)
    producer = {t: op for op, outputs in enumerate(problem["outputs"]) for t in outputs}
    consumers = [set() for _ in problem["widths"]]
    depth = {}
    edges = set()
    joins = 0
    for op in order:
        parents = {producer[t] for t in problem["inputs"][op] if t in producer}
        depth[op] = 1 + max((depth[parent] for parent in parents), default=0)
        edges.update((parent, op) for parent in parents)
        joins += len(parents) > 1
        for tensor in problem["inputs"][op]:
            consumers[tensor].add(op)
    elements = [w * h for w, h in zip(problem["widths"], problem["heights"])]
    return {
        "nodes": len(order), "dependency_edges": len(edges), "tensors": len(elements),
        "depth": max(depth.values()), "matmul_ops": problem["op_types"].count("MatMul"),
        "pointwise_ops": problem["op_types"].count("Pointwise"), "join_ops": joins,
        "fanout_tensors": sum(len(c) > 1 for c in consumers),
        "graph_inputs": len(elements) - len(producer),
        "graph_outputs": sum(not c for c in consumers),
        "min_tensor_elements": min(elements), "max_tensor_elements": max(elements),
        "total_tensor_elements": sum(elements),
        "fast_memory_capacity": problem["fast_memory_capacity"],
        "slow_memory_bandwidth": problem["slow_memory_bandwidth"],
        "native_width": problem["native_granularity"][0],
        "native_height": problem["native_granularity"][1],
    }
