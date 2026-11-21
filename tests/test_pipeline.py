"""Integration checks use temporary copies, AmberTools, and a fake sbatch."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(shutil.which(name) for name in ("pdb4amber", "tleap")):
            raise unittest.SkipTest("Activate an environment containing AmberTools.")
        cls.examples = sorted((ROOT / "structures").rglob("*.pdb"))
        if len(cls.examples) < 2:
            raise unittest.SkipTest("At least two example AF2 PDBs are needed for integration checks.")
        cls.fixture = tempfile.TemporaryDirectory(prefix="automated_meld_test_")
        cls.base = Path(cls.fixture.name)
        cls.models = cls.base / "input models"
        (cls.models / "first").mkdir(parents=True)
        (cls.models / "second").mkdir()
        # Same basename in different directories must produce distinct runs.
        shutil.copy2(cls.examples[0], cls.models / "first" / "model with spaces.pdb")
        shutil.copy2(cls.examples[1], cls.models / "second" / "model with spaces.pdb")
        config = json.loads((ROOT / "templates" / "input.json").read_text())
        config["n_replicas"] = 2
        cls.config = cls.base / "test_config.json"
        cls.config.write_text(json.dumps(config))
        cls.prepared = cls.base / "prepared fixture"
        result = cls.command("pipeline.py", "--structures", cls.models, "--output", cls.prepared,
                             "--config", cls.config)
        if result.returncode:
            cls.fixture.cleanup()
            raise RuntimeError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    @staticmethod
    def command(script, *args, env=None):
        return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                              cwd=ROOT, env=env, capture_output=True, text=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="automated_meld_case_")
        self.area = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def copied_output(self):
        destination = self.area / "relocated simulations"
        shutil.copytree(self.prepared, destination)
        return destination

    def test_variable_models_names_and_unchanged_resume(self):
        output = self.copied_output()
        manifest = json.loads((output / "pipeline_manifest.json").read_text())
        self.assertEqual(len(manifest["runs"]), 2)
        self.assertEqual(len({r["id"] for r in manifest["runs"]}), 2)
        for run in manifest["runs"]:
            folder = output / run["directory"]
            self.assertTrue((folder / "sequence.fasta").is_file())
            self.assertFalse((folder / "Data").exists())
            self.assertFalse((folder / "Logs").exists())
            self.assertEqual((folder / "raw_af2.pdb").read_bytes(),
                             (self.models / run["source_relative"]).read_bytes())
        result = self.command("pipeline.py", "--structures", self.models, "--output", output,
                              "--config", self.config, "--resume")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Already complete", result.stdout)

    def test_mixed_sequences_rejected_before_preparation(self):
        models = self.area / "mixed"
        models.mkdir()
        shutil.copy2(self.examples[0], models / "a.pdb")
        lines = self.examples[0].read_text().splitlines()
        atom = next(line for line in lines if line.startswith("ATOM  "))
        first_key = atom[21:27]
        replacement = "ALA" if atom[17:20] != "ALA" else "GLY"
        changed = [line[:17] + replacement + line[20:]
                   if line.startswith("ATOM  ") and line[21:27] == first_key else line for line in lines]
        (models / "b.pdb").write_text("\n".join(changed) + "\n")
        output = self.area / "bad_output"
        result = self.command("pipeline.py", "--structures", models, "--output", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Sequence mismatch", result.stderr)
        self.assertFalse(output.exists())

    def test_changed_staged_input_rejected(self):
        output = self.copied_output()
        manifest = json.loads((output / "pipeline_manifest.json").read_text())
        sequence = output / manifest["runs"][0]["directory"] / "sequence.dat"
        sequence.write_text("A\n")
        result = self.command("pipeline.py", "--structures", self.models, "--output", output,
                              "--config", self.config, "--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Staged input changed", result.stderr)

    def test_submission_guard_and_duplicate_prevention_with_fake_scheduler(self):
        output = self.copied_output()
        fake = self.area / "fake commands"
        fake.mkdir()
        log = self.area / "fake_sbatch_calls.jsonl"
        scheduler = fake / "sbatch"
        scheduler.write_text('''#!/usr/bin/env python3
import json, os
from pathlib import Path
p = Path(os.environ["FAKE_SBATCH_LOG"])
n = len(p.read_text().splitlines()) if p.exists() else 0
with p.open("a") as f: f.write(json.dumps({"cwd": os.getcwd()}) + "\\n")
print(1001 + n)
''')
        scheduler.chmod(0o755)
        env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ["PATH"], FAKE_SBATCH_LOG=str(log))
        blocked = self.command("submit_jobs.py", "--simulations", output, "--submit", env=env)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertFalse(log.exists())
        for job in output.glob("model_*/job.slurm"):
            contents = job.read_text()
            for word in ("YOUR_ACCOUNT", "YOUR_GPU_PARTITION", "YOUR_CONDA_MODULE", "YOUR_CUDA_MODULE"):
                contents = contents.replace(word, "example_site_value")
            job.write_text(contents)
        preview = self.command("submit_jobs.py", "--simulations", output, env=env)
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertFalse(log.exists())
        submitted = self.command("submit_jobs.py", "--simulations", output, "--submit", env=env)
        self.assertEqual(submitted.returncode, 0, submitted.stderr)
        self.assertEqual(len(log.read_text().splitlines()), 2)
        repeated = self.command("submit_jobs.py", "--simulations", output, "--submit", env=env)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(len(log.read_text().splitlines()), 2)
        rows = [json.loads(line) for line in (output / "submissions.jsonl").read_text().splitlines()]
        self.assertEqual([row["job_id"] for row in rows], ["1001", "1002"])

    def test_single_model_empty_collections_and_setup_only(self):
        if importlib.util.find_spec("meld") is None:
            self.skipTest("MELD is needed for the serialized setup checks.")
        models = self.area / "one model"
        models.mkdir()
        # Fixed confidence makes the empty-core/empty-alternative cases deterministic.
        lines = self.examples[0].read_text().splitlines()
        (models / "one.pdb").write_text("\n".join(
            line[:60] + " 90.00" + line[66:] if line.startswith("ATOM  ") else line
            for line in lines) + "\n")
        for threshold, empty_collection in ((0, "alternatives"), (100, "core")):
            config = json.loads(self.config.read_text())
            config["contact_generation"]["minimum_core_plddt"] = threshold
            config_path = self.area / f"config_{threshold}.json"
            config_path.write_text(json.dumps(config))
            output = self.area / f"single_{threshold}"
            staged = self.command("pipeline.py", "--structures", models, "--output", output,
                                  "--config", config_path)
            self.assertEqual(staged.returncode, 0, staged.stdout + staged.stderr)
            manifest = json.loads((output / "pipeline_manifest.json").read_text())
            run = output / manifest["runs"][0]["directory"]
            summary = json.loads((output / "reports" / "contact_summary.json").read_text())
            self.assertEqual(summary[empty_collection + "_contacts" if empty_collection == "core" else "alternative_contacts"], 0)
            built = subprocess.run([sys.executable, str(run / "system_setup.py")], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            report = json.loads((run / "setup_report.json").read_text())
            self.assertEqual(report["dynamics_steps_executed"], 0)
            self.assertEqual(report["restraints"][empty_collection]["total_contacts"], 0)
            self.assertTrue((run / "Data").is_dir())
            self.assertTrue((run / "Logs").is_dir())


if __name__ == "__main__":
    unittest.main()
