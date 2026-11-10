"""Tests for i2r.runio and the run.py skeleton. No GPU, no network."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from i2r import runio
from i2r.runio import ContractError, require_keys


class RequireKeysTest(unittest.TestCase):
    def test_passes_when_all_keys_present(self) -> None:
        require_keys({"a": 1, "b": 2}, ("a", "b"), where="x.json")

    def test_names_file_and_missing_key(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            require_keys({"a": 1}, ("a", "x_range"), where="bays.json")
        message = str(ctx.exception)
        self.assertIn("bays.json", message)
        self.assertIn("x_range", message)

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(ContractError):
            require_keys([1, 2], ("a",), where="list.json")


class JsonTest(unittest.TestCase):
    def test_round_trip_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "a.json"
            runio.write_json(path, {"ç": [1, 2]})
            self.assertEqual(runio.read_json(path), {"ç": [1, 2]})
            with self.assertRaises(ContractError):
                runio.read_json(Path(tmp) / "missing.json")


class CreateRunTest(unittest.TestCase):
    """Uses a temporary data and output root through a temporary config file."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        cfg = runio.load_config()
        cfg["paths"]["data_root"] = str(self.tmp / "data")
        cfg["paths"]["output_root"] = str(self.tmp / "out")
        (self.tmp / "data" / "demo").mkdir(parents=True)
        self.config_path = self.tmp / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_folder_meta_and_config_copy(self) -> None:
        run = runio.create_run("demo", self.config_path)
        try:
            self.assertTrue(run.run_dir.is_dir())
            self.assertEqual(run.run_dir.parent.name, "demo")
            self.assertTrue((run.run_dir / "config.yaml").is_file())
            meta = runio.read_json(run.run_dir / "run_meta.json")
            require_keys(meta, ("run_id", "environment", "git_commit", "status"), "run_meta.json")
            self.assertIn("executable", meta["environment"])
        finally:
            runio.close_logger(run)

    def test_unknown_structure_lists_known_ones(self) -> None:
        with self.assertRaises(ContractError) as ctx:
            runio.create_run("nope", self.config_path)
        self.assertIn("demo", str(ctx.exception))

    def test_run_py_logs_all_six_steps(self) -> None:
        import run as entry

        rc = entry.main(["--structure", "demo", "--no-review", "--config", str(self.config_path)])
        self.assertEqual(rc, 0)
        run_dir = next((self.tmp / "out" / "demo").iterdir())
        log = (run_dir / "run.log").read_text(encoding="utf-8")
        for step in runio.STEPS:
            self.assertIn(f"step: {step}", log)
        meta = runio.read_json(run_dir / "run_meta.json")
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(list(meta["steps"]), list(runio.STEPS))


if __name__ == "__main__":
    unittest.main()
