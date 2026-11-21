"""Extraction tests run a fake extract_trajectory, never MELD or a real trajectory."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "templates" / "extract_trajectories.sh"


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="meld_extraction_test_")
        self.area = Path(self.temporary.name)
        self.fake = self.area / "fake commands"
        self.fake.mkdir()
        self.calls = self.area / "calls.jsonl"
        extractor = self.fake / "extract_trajectory"
        extractor.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
assert sys.argv[1:3] == ['extract_traj_dcd', '--replica']
replica = int(sys.argv[3])
with Path(os.environ['EXTRACTION_TEST_CALLS']).open('a') as handle:
    handle.write(json.dumps({'replica': replica, 'cwd': os.getcwd()}) + '\\n')
Path(sys.argv[4]).write_text('fake DCD for testing replica ' + str(replica))
if os.environ.get('EXTRACTION_TEST_FAIL') == str(replica):
    sys.exit(1)
''')
        extractor.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.fake) + os.pathsep + os.environ["PATH"],
                        EXTRACTION_TEST_CALLS=str(self.calls))

    def tearDown(self):
        self.temporary.cleanup()

    def model(self, name="model with spaces", count=3, data=True):
        folder = self.area / name
        folder.mkdir()
        (folder / "input.json").write_text(json.dumps({"n_replicas": count}))
        shutil.copy2(SCRIPT, folder / "extract_trajectories.sh")
        if data:
            (folder / "Data").mkdir()
            (folder / "Data" / "data_store.dat").touch()
        return folder

    def extract(self, folder, *args, env=None):
        return subprocess.run(["bash", str(SCRIPT), "--run-dir", str(folder), *args],
                              env=env or self.env, capture_output=True, text=True)

    def test_default_indices_and_existing_output_skip(self):
        folder = self.model()
        (folder / "1.dcd").write_text("keep existing")
        result = self.extract(folder)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line)["replica"] for line in self.calls.read_text().splitlines()]
        self.assertEqual(calls, [0, 2])
        self.assertEqual((folder / "1.dcd").read_text(), "keep existing")
        self.assertTrue((folder / "0.dcd").is_file())
        self.assertTrue((folder / "2.dcd").is_file())
        self.assertFalse((folder / "3.dcd").exists())

    def test_dry_run_and_out_of_range(self):
        folder = self.model(count=31, data=False)
        before = {p.name for p in folder.iterdir()}
        result = self.extract(folder, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--replica 30 30.dcd", result.stdout)
        self.assertEqual(before, {p.name for p in folder.iterdir()})
        self.assertFalse(self.calls.exists())
        invalid = self.extract(folder, "--last-replica", "31", "--dry-run")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("indexed 0-30", invalid.stderr)

    def test_failed_overwrite_preserves_original_and_cleans_temporary(self):
        folder = self.model()
        (folder / "0.dcd").write_text("original")
        result = self.extract(folder, "--last-replica", "0", "--overwrite",
                              env=dict(self.env, EXTRACTION_TEST_FAIL="0"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((folder / "0.dcd").read_text(), "original")
        self.assertFalse(list(folder.glob(".*.extracting.*.dcd")))

    def test_batch_uses_manifest_and_writes_summary(self):
        project = self.area / "simulations"
        project.mkdir()
        runs = []
        for index in range(2):
            folder = self.model(name=f"source_{index}", count=2)
            destination = project / f"model_{index}"
            shutil.move(str(folder), destination)
            runs.append({"id": destination.name, "directory": destination.name})
        (project / "pipeline_manifest.json").write_text(json.dumps({"status": "complete", "runs": runs}))
        result = subprocess.run([sys.executable, str(ROOT / "extract_all_trajectories.py"),
                                 "--simulations", str(project)], env=self.env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((project / "extraction_summary.json").read_text())
        self.assertTrue(summary["all_passed"])
        self.assertEqual(len(summary["results"]), 2)
        self.assertEqual(len(self.calls.read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
