// Project-owned C ABI. Upstream VNRC remains unmodified and separately licensed.
// Python performs schema/state validation before calling this private bridge.
#include "core/cost.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

using json = nlohmann::json;
namespace {
constexpr const char* revision = "1f19c2ffeae394ab7c5f1d92c06795debf667758";
char* encode(const json& value) {
    const auto text = value.dump();
    auto* memory = static_cast<char*>(std::malloc(text.size() + 1));
    if (!memory) return nullptr;
    std::memcpy(memory, text.c_str(), text.size() + 1);
    return memory;
}
int64_t integer(const json& value, int64_t low, int64_t high) {
    if (!value.is_number_integer()) throw std::runtime_error("expected integer");
    if (value.is_number_unsigned() && value.get<uint64_t>() > static_cast<uint64_t>(high))
        throw std::runtime_error("integer out of range");
    const auto n = value.get<int64_t>();
    if (n < low || n > high) throw std::runtime_error("integer out of range");
    return n;
}
struct Context { Problem problem; DAG dag; };
Problem parse_problem(const json& j) {
    Problem p;
    const auto& widths = j.at("widths");
    const auto& heights = j.at("heights");
    if (!widths.is_array() || widths.empty() || widths.size() != heights.size())
        throw std::runtime_error("invalid tensor arrays");
    for (size_t i = 0; i < widths.size(); ++i)
        p.tensors.push_back({integer(widths.at(i), 1, INT32_MAX),
                             integer(heights.at(i), 1, INT32_MAX)});
    const auto& types = j.at("op_types");
    if (!types.is_array() || types.empty()) throw std::runtime_error("empty operations");
    for (auto key : {"inputs", "outputs", "base_costs"})
        if (!j.at(key).is_array() || j.at(key).size() != types.size())
            throw std::runtime_error("invalid operation arrays");
    std::vector<bool> produced(p.tensors.size());
    for (size_t i = 0; i < types.size(); ++i) {
        const auto kind = types.at(i).get<std::string>();
        if (kind != "MatMul" && kind != "Pointwise") throw std::runtime_error("invalid operation type");
        Op op;
        op.type = kind == "MatMul" ? OpType::MatMul : OpType::Pointwise;
        for (const auto& t : j.at("inputs").at(i))
            op.inputs.push_back(integer(t, 0, p.tensors.size() - 1));
        if (op.inputs.empty() || j.at("outputs").at(i).size() != 1)
            throw std::runtime_error("unsupported operation arity");
        const size_t t = integer(j.at("outputs").at(i).at(0), 0, p.tensors.size() - 1);
        if (produced[t]) throw std::runtime_error("multiple producers");
        produced[t] = true;
        op.outputs.push_back(t);
        op.base_cost = integer(j.at("base_costs").at(i), 1, INT64_MAX);
        p.ops.push_back(std::move(op));
    }
    p.fast_memory_capacity = integer(j.at("fast_memory_capacity"), 1, INT64_MAX);
    p.slow_memory_bandwidth = integer(j.at("slow_memory_bandwidth"), 1, INT64_MAX);
    p.native_w = integer(j.at("native_granularity").at(0), 1, INT32_MAX);
    p.native_h = integer(j.at("native_granularity").at(1), 1, INT32_MAX);
    return p;
}
}

extern "C" {
int evaluator_abi_version() { return 1; }
const char* evaluator_revision() { return revision; }
void evaluator_free(void* memory) { std::free(memory); }
void evaluator_destroy(void* handle) { delete static_cast<Context*>(handle); }
void* evaluator_create(const char* problem, char** error) {
    *error = nullptr;
    try {
        auto context = std::make_unique<Context>();
        context->problem = parse_problem(json::parse(problem));
        context->dag = DAG::build(context->problem);
        if (context->dag.topological_order().size() != context->problem.num_ops())
            throw std::runtime_error("cyclic problem");
        return context.release();
    } catch (const std::exception& e) {
        *error = encode({{"error", e.what()}});
        return nullptr;
    }
}
char* evaluator_score(void* handle, const char* schedule) {
    try {
        if (!handle) throw std::runtime_error("closed evaluator");
        const auto& context = *static_cast<Context*>(handle);
        const auto& p = context.problem;
        const auto request = json::parse(schedule);
        FlatSet<size_t> entering;
        json latencies = json::array(), working_sets = json::array();
        double total = 0;
        size_t index = 0;
        for (const auto& step : request.at("steps")) {
            std::vector<size_t> ops;
            FlatSet<size_t> unique;
            for (const auto& op : step.at("ops")) {
                size_t i = integer(op, 0, p.num_ops() - 1);
                if (!unique.insert(i).second) throw std::runtime_error("duplicate operation");
                ops.push_back(i);
            }
            auto graph = Subgraph::create(p, context.dag, std::move(ops));
            if (!graph) return encode({{"valid", false}, {"code", "unsupported_subgraph"}, {"step", index}});
            const auto& gran = step.at("granularity");
            TileConfig cfg{integer(gran.at(0), 1, INT32_MAX),
                           integer(gran.at(1), 1, INT32_MAX),
                           integer(gran.at(2), 1, INT32_MAX),
                           static_cast<SnakeDir>(integer(step.at("traversal"), 0, 2))};
            FlatSet<size_t> retain;
            for (const auto& tensor : step.at("retain"))
                retain.insert(integer(tensor, 0, p.num_tensors() - 1));
            if (!graph->is_valid_tiling(cfg))
                return encode({{"valid", false}, {"code", "invalid_tiling"}, {"step", index}});
            const auto cost = graph->compute_cost(cfg, entering, retain);
            if (!cost.feasible)
                return encode({{"valid", false}, {"code", "out_of_memory"}, {"step", index}});
            if (!std::isfinite(cost.latency) || cost.latency <= 0 || !std::isfinite(total + cost.latency))
                return encode({{"valid", false}, {"code", "nonfinite_cost"}, {"step", index}});
            total += cost.latency;
            latencies.push_back(cost.latency);
            working_sets.push_back(cost.working_set);
            entering = std::move(retain);
            ++index;
        }
        return encode({{"valid", true}, {"total_latency", total},
                       {"subgraph_latencies", latencies}, {"working_sets", working_sets}});
    } catch (const std::exception& e) {
        return encode({{"valid", false}, {"code", "backend_error"}, {"error", e.what()}});
    }
}
}
