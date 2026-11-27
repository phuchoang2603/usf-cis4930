from random import Random


DIMENSIONS = (128, 192, 256, 384, 512, 768, 1024)


class Graph:
    def __init__(self, rng: Random):
        self.rng = rng
        self.shapes = []
        self.inputs = []
        self.outputs = []
        self.costs = []
        self.kinds = []

    def tensor(self, height, width):
        index = len(self.shapes)
        self.shapes.append((height, width))
        return index

    def operation(self, kind, inputs, shape, costs):
        out = self.tensor(*shape)
        self.inputs.append(list(inputs))
        self.outputs.append([out])
        self.costs.append(self.rng.choice(costs))
        self.kinds.append(kind)
        return out

    def matmul(self, source, width):
        height, reduction = self.shapes[source]
        weight = self.tensor(reduction, width)
        return self.operation("MatMul", (source, weight), (height, width), (1000, 2000, 3000, 5000))

    def pointwise(self, source, *others):
        return self.operation("Pointwise", (source, *others), self.shapes[source], (100, 200, 300, 500))

    def projection(self, source):
        return self.pointwise(self.matmul(source, self.rng.choice(DIMENSIONS)))

    def residual(self, source):
        width = self.shapes[source][1]
        expanded = self.pointwise(self.matmul(source, 2 * width))
        return self.pointwise(self.matmul(expanded, width), source)

    def gated(self, source):
        width = self.shapes[source][1]
        value = self.matmul(source, 2 * width)
        gate = self.pointwise(self.matmul(source, 2 * width))
        return self.matmul(self.pointwise(value, gate), width)

    def fork_join(self, source):
        width = self.rng.choice(DIMENSIONS)
        return self.pointwise(self.matmul(source, width), self.matmul(source, width))

    def problem(self, capacity, bandwidth):
        return dict(
            widths=[width for _, width in self.shapes],
            heights=[height for height, _ in self.shapes],
            inputs=self.inputs, outputs=self.outputs, base_costs=self.costs, op_types=self.kinds,
            fast_memory_capacity=capacity, slow_memory_bandwidth=bandwidth, native_granularity=[128, 128],
        )


MOTIFS = (Graph.projection, Graph.residual, Graph.gated, Graph.fork_join)


def generate_problem(motifs, rng: Random, capacity, bandwidth):
    graph = Graph(rng)
    output = graph.tensor(rng.choice(DIMENSIONS), rng.choice(DIMENSIONS))
    for motif in motifs:
        output = motif(graph, output)
    return graph.problem(capacity, bandwidth)
