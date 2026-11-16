"""Competition input schema and graph validation, independent of data generation."""
from collections import deque


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
