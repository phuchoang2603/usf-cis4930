import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluator import Evaluator
from scripts.generate_synthetic import write_dataset


ROOT = Path(__file__).resolve().parents[1]


class SyntheticTests(unittest.TestCase):
    def test_cli_reproduces_disjoint_evaluable_splits(self):
        with TemporaryDirectory() as directory:
            outputs = [Path(directory) / name for name in ("first", "second")]
            for output in outputs:
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.generate_synthetic", "--output", str(output),
                     "--seed", "11", "--train", "6", "--validation", "3", "--test", "3"],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            manifest = json.loads((outputs[0] / "manifest.json").read_text())
            self.assertEqual(manifest, json.loads((outputs[1] / "manifest.json").read_text()))
            self.assertEqual({split: len(rows) for split, rows in manifest["splits"].items()},
                             dict(train=6, validation=3, test=3))
            topologies = set()
            for rows in manifest["splits"].values():
                self.assertEqual({motif for row in rows for motif in row["motifs"]},
                                 {"projection", "residual", "gated", "fork_join"})
                self.assertGreaterEqual(len({len(row["motifs"]) for row in rows}), 3)
                for row in rows:
                    path = row["path"]
                    raw = (outputs[0] / path).read_bytes()
                    self.assertEqual(raw, (outputs[1] / path).read_bytes())
                    problem = json.loads(raw)
                    producers = {out[0]: op for op, out in enumerate(problem["outputs"])}
                    topology = tuple((kind, tuple(producers.get(tensor, -1) for tensor in inputs))
                                     for kind, inputs in zip(problem["op_types"], problem["inputs"]))
                    self.assertNotIn(topology, topologies)
                    topologies.add(topology)
                    consumed = {tensor for inputs in problem["inputs"] for tensor in inputs}
                    self.assertEqual(set(producers) - consumed, {problem["outputs"][-1][0]})
                    with Evaluator(problem) as evaluator:
                        baseline = evaluator.baseline()
                        self.assertTrue(evaluator.evaluate(baseline["schedule"], submission=True)["valid"])

    def test_baseline_failure_leaves_no_partial_dataset(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            with patch.object(Evaluator, "baseline", side_effect=({}, RuntimeError("cannot score"))):
                with self.assertRaisesRegex(ValueError, "train/00001.json: cannot score"):
                    write_dataset(output, 11, dict(train=2))
            self.assertEqual(list(Path(directory).iterdir()), [])
