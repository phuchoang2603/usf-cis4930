// Benchmark adapter only. Links the unmodified, pinned VNRC cost model.
// Prepared scoring excludes parsing, DAG/subgraph construction, coverage/order
// validation, diagnostics, and the upstream CLI's best_cost() searches.
#include "core/cost.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "io/io.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

struct Step {
    Subgraph graph;
    TileConfig config;
    FlatSet<size_t> retain;
};

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: driver problem.json solution.json");
        const auto start = Clock::now();
        auto problem = read_problem(argv[1]);
        auto dag = DAG::build(problem);
        std::ifstream file(argv[2]);
        const json schedule = json::parse(file);
        std::vector<Step> steps;
        for (size_t i = 0; i < schedule.at("subgraphs").size(); ++i) {
            auto graph = Subgraph::create(problem, dag,
                schedule.at("subgraphs").at(i).get<std::vector<size_t>>());
            if (!graph) throw std::runtime_error("unsupported subgraph");
            const auto& gran = schedule.at("granularities").at(i);
            TileConfig cfg{gran.at(0).get<int64_t>(), gran.at(1).get<int64_t>(),
                           gran.at(2).get<int64_t>(), SnakeDir::RowMajor};
            if (cfg.w <= 0 || cfg.h <= 0 || cfg.k <= 0)
                throw std::runtime_error("nonpositive granularity");
            // Reproduce upstream CLI semantics for an apples-to-apples total.
            if (schedule.contains("traversal_orders") &&
                !schedule.at("traversal_orders").at(i).is_null() &&
                !schedule.at("traversal_orders").at(i).empty()) {
                const auto order = schedule.at("traversal_orders").at(i).get<std::vector<int>>();
                const int nw = std::max<int64_t>(graph->output_width() / cfg.w, 1);
                const int nh = std::max<int64_t>(graph->output_height() / cfg.h, 1);
                if (order == make_traversal(nw, nh, SnakeDir::RowMajor))
                    cfg.snake = SnakeDir::RowMajor;
                else if (order == make_traversal(nw, nh, SnakeDir::ColMajor))
                    cfg.snake = SnakeDir::ColMajor;
                else
                    cfg.snake = SnakeDir::None;
            }
            FlatSet<size_t> retain;
            for (auto& t : schedule.at("tensors_to_retain").at(i))
                retain.insert(t.get<size_t>());
            steps.push_back({std::move(*graph), cfg, std::move(retain)});
        }
        const auto prepared = Clock::now();
        auto score = [&]() {
            FlatSet<size_t> entering;
            double total = 0;
            for (const auto& step : steps) {
                auto cost = step.graph.compute_cost(step.config, entering, step.retain);
                if (!cost.feasible || !std::isfinite(cost.latency))
                    throw std::runtime_error("infeasible schedule");
                total += cost.latency;
                entering = step.retain;
            }
            return total;
        };
        const double reference = score();
        volatile double checksum = 0;
        std::vector<double> samples;
        std::vector<size_t> iterations;
        for (int trial = 0; trial < 7; ++trial) {
            const auto begin = Clock::now();
            size_t count = 0;
            Clock::time_point end;
            do {
                checksum = checksum + score();
                ++count;
                end = Clock::now();
            } while (end - begin < std::chrono::milliseconds(10) && count < 100000);
            samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count() / count);
            iterations.push_back(count);
        }
        std::cout << json({{"setup_us", std::chrono::duration<double, std::micro>(prepared - start).count()},
                           {"prepared_score_us", samples}, {"iterations", iterations},
                           {"total", reference}, {"steps", steps.size()},
                           {"checksum", checksum}}).dump() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
