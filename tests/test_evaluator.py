import copy
import contextlib
import io
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from evaluator import Evaluator, MODEL_ID, REWARD_ID

ROOT = Path(__file__).resolve().parents[1]


def chain():
    return dict(widths=[128] * 3, heights=[128] * 3, inputs=[[0], [1]], outputs=[[1], [2]],
                base_costs=[1000, 100], op_types=["Pointwise"] * 2,
                fast_memory_capacity=100000, slow_memory_bandwidth=10,
                native_granularity=[128, 128])


def plan(groups=None, retain=None):
    groups = [[0], [1]] if groups is None else groups
    return dict(subgraphs=groups, granularities=[[128, 128, 1] for _ in groups],
                tensors_to_retain=retain if retain is not None else [[] for _ in groups])


def official_examples():
    text = (ROOT / "PROBLEM.md").read_text().split("## Examples", 1)[1]
    problem, label = None, ""
    for match in re.finditer(r"^### (Example[^\n]+)|^#### (Strategy[^\n]+)|```json\s*\n(.*?)```",
                             text, re.M | re.S):
        if match[1] or match[2]:
            label = match[1] or match[2]
        elif match[3].strip() != "NA":
            value = json.loads(re.sub(r"//[^\n]*", "", match[3]))
            if "widths" in value:
                problem = value
            elif "subgraphs" in value:
                yield label, problem, value


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator(chain())
        self.addCleanup(self.ev.close)

    def invalid(self, schedule, code=None, **kwargs):
        result = self.ev.evaluate(schedule, **kwargs)
        self.assertFalse(result["valid"], result)
        self.assertIsNone(result["total_latency"])
        if code:
            self.assertEqual(result["code"], code)
        json.dumps(result, allow_nan=False)
        return result

    def test_all_official_examples_without_traversal_rewriting(self):
        examples = list(official_examples())
        self.assertEqual(len(examples), 11)
        for label, problem, schedule in examples:
            with self.subTest(label=label), Evaluator(problem, allow_snake=True) as ev:
                result = ev.evaluate(schedule, submission=True)
                self.assertTrue(result["valid"], result)
                self.assertAlmostEqual(result["total_latency"], sum(schedule["subgraph_latencies"]), places=3)
                self.assertTrue(result["reported_latencies_match"])

    def test_default_raster_and_explicit_snake(self):
        examples = list(official_examples())
        p, raster = examples[8][1:]
        snake = examples[9][2]
        with Evaluator(p) as ev:
            self.assertAlmostEqual(ev.evaluate(raster)["total_latency"], 7096)
            self.assertEqual(ev.evaluate(snake)["code"], "unsupported_traversal")
            omitted = copy.deepcopy(raster)
            del omitted["traversal_orders"]
            self.assertAlmostEqual(ev.evaluate(omitted)["total_latency"], 7096)
        with Evaluator(p, allow_snake=True) as ev:
            self.assertAlmostEqual(ev.evaluate(snake)["total_latency"], 6548)
            for order in ([0, 0, 0, 0], [0, 1], [0, 1, 2, 4], [0, 3, 1, 2], []):
                s = copy.deepcopy(raster)
                s["traversal_orders"] = [order]
                self.assertFalse(ev.evaluate(s)["valid"], order)

    def test_column_snake(self):
        p, s = list(official_examples())[8][1:]
        s = copy.deepcopy(s)
        s["traversal_orders"] = [[0, 2, 3, 1]]
        with Evaluator(p, allow_snake=True) as ev:
            self.assertAlmostEqual(ev.evaluate(s)["total_latency"], 6548)

    def test_consumer_before_producer(self):
        self.invalid(plan([[1], [0]]), "unavailable_tensor")

    def test_duplicate_negative_boolean_and_out_of_range_ops(self):
        for group in ([0, 0], [-1], [True], [2], []):
            with self.subTest(group=group):
                self.invalid(plan([group, [1]]))

    def test_uncovered_operation(self):
        self.invalid(plan([[0]]), "incomplete_coverage")

    def test_invalid_retention(self):
        for retain in ([[0], []], [[2], []], [[1, 1], []], [[99], []], [[True], []]):
            with self.subTest(retain=retain):
                self.invalid(plan(retain=retain))

    def test_final_output_must_reach_slow_memory(self):
        self.invalid(plan(retain=[[], [2]]), "missing_graph_output")

    def test_retain_intermediate(self):
        result = self.ev.evaluate(plan(retain=[[1], []]))
        self.assertTrue(result["valid"], result)
        self.assertAlmostEqual(result["total_latency"], 3276.8)

    def test_recomputation_across_steps_is_allowed(self):
        result = self.ev.evaluate(plan([[0], [0], [1]]))
        self.assertTrue(result["valid"], result)
        self.assertAlmostEqual(result["total_latency"], 9830.4)

    def test_unavailable_ephemeral_and_retained_ephemeral(self):
        p = dict(chain(), widths=[128] * 4, heights=[128] * 4, inputs=[[0], [1], [1]],
                 outputs=[[1], [2], [3]], base_costs=[100] * 3, op_types=["Pointwise"] * 3)
        with Evaluator(p) as ev:
            result = ev.evaluate(plan([[0, 1], [2]]))
            self.assertEqual(result["code"], "unavailable_tensor")
            self.assertTrue(ev.evaluate(plan([[0, 1], [2]], [[1], []]))["valid"])
            # A previously materialized copy remains available through recomputation.
            self.assertTrue(ev.evaluate(plan([[0], [0, 1], [2]]))["valid"])

    def test_dropped_retained_tensor_is_not_implicitly_materialized(self):
        p = dict(chain(), widths=[128] * 6, heights=[128] * 6,
                 inputs=[[0], [1], [1], [4]], outputs=[[1], [2], [3], [5]],
                 base_costs=[100] * 4, op_types=["Pointwise"] * 4)
        with Evaluator(p) as ev:
            result = ev.evaluate(plan([[0, 1], [3], [2]], [[1], [], []]))
            self.assertEqual(result["code"], "unavailable_tensor")

    def test_disconnected_subgraph_rejected(self):
        p = dict(chain(), widths=[128] * 4, heights=[128] * 4,
                 inputs=[[0], [2]], outputs=[[1], [3]])
        with Evaluator(p) as ev:
            self.assertEqual(ev.evaluate(plan([[0, 1]]))["code"], "unsupported_subgraph")

    def test_shared_input_group_supported(self):
        p = dict(chain(), inputs=[[0], [0]])
        with Evaluator(p) as ev:
            self.assertTrue(ev.evaluate(plan([[0, 1]]))["valid"])

    def test_schema_and_nonfinite_reported_latencies(self):
        self.invalid(dict(plan(), subgraph_latencies=None), "invalid_schema")
        for key in ("granularities", "tensors_to_retain", "traversal_orders", "subgraph_latencies"):
            s = plan()
            s[key] = [None]
            self.invalid(s, "invalid_schema")
        for value in (float("nan"), float("inf"), -1, True, 10**1000, "100"):
            s = plan()
            s["subgraph_latencies"] = [value, 3276.8]
            self.invalid(s, "invalid_schema")
        for value in (None, [], "schedule", 4):
            self.invalid(value, "invalid_schema")

    def test_tiles_checked_before_entering_native_code(self):
        for gran in ([0, 128, 1], [-1, 128, 1], [True, 128, 1], [128.0, 128, 1],
                     [63, 128, 1], [256, 128, 1], [128, 128], [2**40, 128, 1]):
            s = plan()
            s["granularities"][0] = gran
            self.invalid(s, "invalid_tiling")

    def test_oom_returns_valid_json_and_finite_reward(self):
        with Evaluator(dict(chain(), fast_memory_capacity=100)) as ev:
            result = ev.evaluate(plan())
            self.assertEqual(result["code"], "out_of_memory")
            self.assertIsNone(result["total_latency"])
            self.assertEqual(ev.reward(plan())["reward"], -1)
            json.dumps(result, allow_nan=False)
            self.assertTrue(ev.baseline()["evaluation"]["valid"])

    def test_submission_labels_are_verified_but_rl_rewards_recompute(self):
        self.invalid(plan(), "invalid_schema", submission=True)
        wrong = dict(plan(), subgraph_latencies=[1, 1])
        self.invalid(wrong, "latency_mismatch", submission=True)
        result = self.ev.evaluate(wrong)
        self.assertTrue(result["valid"])
        self.assertFalse(result["reported_latencies_match"])
        self.assertAlmostEqual(result["total_latency"], 6553.6)
        self.assertEqual(self.ev.reward(wrong)["reward"], self.ev.reward(plan())["reward"])

    def test_baseline_and_bounded_reward(self):
        baseline = self.ev.baseline()
        self.assertTrue(self.ev.evaluate(baseline["schedule"], submission=True)["valid"])
        self.assertEqual(self.ev.reward(baseline["schedule"])["reward"], 0.5)
        self.assertGreater(self.ev.reward(plan([[0, 1]]))["reward"], 0.5)
        self.assertGreater(self.ev.reward(plan([[0], [0], [1]]))["reward"], 0)
        self.assertLess(self.ev.reward(plan([[1], [0]]))["reward"], 0)
        for penalty in (0, 1, float("nan"), float("-inf"), True):
            with self.assertRaises(ValueError):
                self.ev.reward(plan(), invalid_reward=penalty)
        baseline["schedule"]["subgraphs"].clear()
        self.assertTrue(self.ev.baseline()["schedule"]["subgraphs"])

    def test_problem_not_mutated_and_context_lifetime(self):
        p = chain()
        ev = Evaluator(p)
        p["base_costs"][0] = 1
        self.assertEqual(ev.problem["base_costs"][0], 1000)
        ev.close()
        ev.close()
        with self.assertRaises(RuntimeError):
            ev.evaluate(plan())

    def test_configuration_and_checkpoint_model_guard(self):
        cfg = self.ev.configuration
        self.assertEqual(cfg["model_id"], MODEL_ID)
        self.assertEqual(cfg["reward_id"], REWARD_ID)
        self.assertEqual(cfg["traversal_policy"], "raster")
        json.dumps(cfg, allow_nan=False)
        with Evaluator(dict(reversed(list(chain().items()))), expected_model_id=MODEL_ID) as same:
            self.assertEqual(cfg["problem_sha256"], same.configuration["problem_sha256"])
        with Evaluator(dict(chain(), base_costs=[2000, 100])) as different:
            self.assertNotEqual(cfg["problem_sha256"], different.configuration["problem_sha256"])
        with self.assertRaises(ValueError):
            Evaluator(chain(), expected_model_id="different-cost-model")
        for value in ("false", None, 1):
            with self.assertRaises(ValueError):
                Evaluator(chain(), allow_snake=value)
        cfg["model_id"] = "mutated"
        self.assertEqual(self.ev.configuration["model_id"], MODEL_ID)
        with self.assertRaises(AttributeError):
            self.ev.allow_snake = True

    def test_backend_faults_stop_training_instead_of_becoming_action_penalties(self):
        faults = [dict(valid=False, code="backend_error", error="native failure"),
                  dict(valid=False, code="nonfinite_cost"),
                  dict(valid="true"),
                  dict(valid=True, total_latency=float("inf"), subgraph_latencies=[1, 1], working_sets=[1, 1]),
                  dict(valid=True, total_latency=1, subgraph_latencies=[1, 1], working_sets=[1, 1])]
        for fault in faults:
            with self.subTest(fault=fault), patch.object(self.ev._lib, "evaluator_score", return_value=None), \
                 patch("evaluator.api._decode", return_value=fault):
                with self.assertRaises(RuntimeError):
                    self.ev.reward(plan())

    def test_cli_distinguishes_backend_failure_from_invalid_action(self):
        from evaluator.__main__ import main
        output = io.StringIO()
        with patch("sys.argv", ["evaluator", str(ROOT / "example_problem.json")]), \
             patch("evaluator.__main__.Evaluator", side_effect=RuntimeError("backend failure")), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(), 1)
        self.assertEqual(json.loads(output.getvalue())["code"], "backend_or_setup_error")

    def test_reusing_one_context_from_multiple_threads(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.ev.reward(plan()), range(32)))
        self.assertTrue(all(r["reward"] == 0.5 for r in results))

    def test_terminal_rl_integration_demo(self):
        from tools.rl_evaluator_demo import demo
        result = demo()
        self.assertEqual(result["results"]["baseline"]["reward"], 0.5)
        self.assertGreaterEqual(result["results"]["fused"]["reward"], 0.5)
        self.assertEqual(result["results"]["invalid_order"]["reward"], -1)

    def test_malformed_problem_rejected(self):
        for update in (dict(widths=[0] * 3), dict(native_granularity=[128, 64]),
                       dict(outputs=[[1, 2], [2]]), dict(inputs=[[99], [1]]),
                       dict(inputs=[[2], [1]]), dict(widths=[True] * 3)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                Evaluator(dict(chain(), **update))

    def test_non_topological_op_ids(self):
        p = dict(chain(), inputs=[[1], [0]], outputs=[[2], [1]])
        with Evaluator(p) as ev:
            self.assertTrue(ev.baseline()["evaluation"]["valid"])
            self.assertTrue(ev.evaluate(plan([[1], [0]]))["valid"])

    def test_pinned_cost_convention_is_explicit(self):
        for k in (128, 256, 512):
            p = dict(chain(), widths=[k, 128, 128], heights=[128, k, 128],
                     inputs=[[0, 1]], outputs=[[2]], base_costs=[2000], op_types=["MatMul"],
                     slow_memory_bandwidth=1000000)
            s = dict(subgraphs=[[0]], granularities=[[128, 128, 64]], tensors_to_retain=[[]])
            with Evaluator(p) as ev:
                result = ev.evaluate(s)
                self.assertTrue(result["valid"])
                self.assertAlmostEqual(result["total_latency"], 2000)
                self.assertEqual(result["model_id"], MODEL_ID)
                self.assertFalse(result["official_equivalence_verified"])

    def test_all_124_workloads_have_feasible_baselines(self):
        paths = sorted((ROOT / "benchmarks").glob("*.json"))
        paths += sorted((ROOT / "data/generated/synthetic").glob("*/*.json"))
        self.assertEqual(len(paths), 124)
        for path in paths:
            with self.subTest(path=path), Evaluator(json.loads(path.read_text())) as ev:
                baseline = ev.baseline()
                self.assertTrue(baseline["evaluation"]["valid"])
                self.assertTrue(ev.evaluate(baseline["schedule"], submission=True)["valid"])
                self.assertEqual(ev.reward(baseline["schedule"])["reward"], 0.5)

    def test_mutated_actions_do_not_crash_or_emit_nonfinite_results(self):
        rng = random.Random(4930)
        mutations = [None, -1, True, {}, [], [1], [0, 0], [128, 128, 0], "invalid"]
        for _ in range(100):
            s = plan()
            key = rng.choice(["subgraphs", "granularities", "tensors_to_retain", "traversal_orders"])
            s[key] = rng.choice(mutations)
            result = self.ev.evaluate(s)
            json.dumps(result, allow_nan=False)

    def test_cli_machine_readable_success_and_failure(self):
        problem = ROOT / "example_problem.json"
        good = subprocess.run([sys.executable, "-m", "evaluator", str(problem)],
                              cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stderr + good.stdout)
        self.assertTrue(json.loads(good.stdout)["evaluation"]["valid"])
        bad = subprocess.run([sys.executable, "-m", "evaluator", "/nonexistent/problem.json"],
                             cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(bad.returncode, 2)
        self.assertFalse(json.loads(bad.stdout)["valid"])


if __name__ == "__main__":
    unittest.main()
