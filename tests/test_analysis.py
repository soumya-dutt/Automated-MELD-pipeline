"""Portable analysis checks with synthetic trajectories; no MELD, GPU or scheduler."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import mdtraj as md
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise unittest.SkipTest("Install the analysis/environment.yml dependencies for analysis tests") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.metrics import assignments, backbone, rmsd_A


@unittest.skipUnless(importlib.util.find_spec("threadpoolctl") and importlib.util.find_spec("matplotlib"),
                     "Activate the analysis environment for analysis integration tests")
class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="portable_meld_analysis_")
        self.area = Path(self.temp.name)
        self.simulations = self.area / "simulations with spaces"
        self.simulations.mkdir()
        self.output = self.area / "analysis"
        self.names = ["first model", "second-model", "third_model"]
        top = md.Topology()
        chain = top.add_chain()
        for _ in range(8):
            residue = top.add_residue("ALA", chain)
            for name, element in [("N", md.element.nitrogen), ("CA", md.element.carbon),
                                  ("C", md.element.carbon), ("O", md.element.oxygen), ("CB", md.element.carbon)]:
                top.add_atom(name, element, residue)
        rng = np.random.default_rng(7)
        a = rng.normal(0, .3, (top.n_atoms, 3))
        self.xyz = [a, a+rng.normal(0, .25, a.shape), a+rng.normal(0, .45, a.shape)]
        records = []
        for index, name in enumerate(self.names):
            folder = self.simulations / f"model_{index}"
            folder.mkdir()
            pdb = folder / "prot_meld_input.pdb"
            md.Trajectory(self.xyz[index][None], top).save_pdb(str(pdb))
            config = {"n_replicas": 2, "run_id": name, "langevin_seed": 100+index}
            (folder / "input.json").write_text(json.dumps(config))
            for replica, count in [(0, 6), (1, 4)]:
                frames = np.stack([self.xyz[j % 3] + [j*.01, 1, 2] for j in range(count)])
                md.Trajectory(frames, top).save_dcd(str(folder / f"{replica}.dcd"))
            records.append({"id": name, "directory": folder.name})
        (self.simulations / "pipeline_manifest.json").write_text(json.dumps({"status": "complete", "runs": records}))

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "analyze_trajectories.py"),
                               "--simulations", str(self.simulations), "--output", str(self.output), *args],
                              capture_output=True, text=True)

    def test_rotation_translation_reflection_and_overlap(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(12, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        self.assertLess(rmsd_A((a@rotation+[2, 3, 4])[None], a)[0], 1e-10)
        reflection = a.copy(); reflection[:, 0] *= -1
        self.assertGreater(rmsd_A(reflection[None], a)[0], 1)
        hits, labels = assignments([[1, 9, 9], [4, 4, 9], [9, 9, 9]], 4)
        self.assertEqual(list(labels), ["reference_0000", "ambiguous", "unassigned"])
        self.assertEqual(hits.sum(axis=1).tolist(), [1, 2, 0])

    def test_backbone_mapping_ignores_atom_order_and_pdb_numbering(self):
        source = self.simulations / "model_0/prot_meld_input.pdb"
        ref, indices, keys, _ = backbone(source)
        lines = source.read_text().splitlines()
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        shuffled = []
        for start in range(0, len(atom_lines), 5):
            for line in reversed(atom_lines[start:start+5]):
                shuffled.append(line[:22] + f"{int(line[22:26])+40:4d}" + line[26:])
        path = self.area / "reordered.pdb"
        path.write_text("\n".join(shuffled) + "\nEND\n")
        other, other_indices, other_keys, _ = backbone(path)
        self.assertEqual(keys, other_keys)
        self.assertNotEqual(indices.tolist(), other_indices.tolist())
        self.assertLess(rmsd_A(other.xyz[:, other_indices], ref.xyz[0, indices])[0], 1e-10)
        trajectories = [md.load(str(source.parent / f"{r}.dcd"), top=str(source)) for r in range(2)]
        permutation = np.concatenate([np.arange(i, i+5)[::-1] for i in range(0, ref.n_atoms, 5)])
        source.write_text(path.read_text())
        for replica, trajectory in enumerate(trajectories):
            md.Trajectory(trajectory.xyz[:, permutation], other.top).save_dcd(str(source.parent / f"{replica}.dcd"))
        result = self.invoke("--replicas", "0")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        table = pd.read_csv(self.output / "reference_0000/frame_rmsd.csv")
        for frame, reference_id in [(0, 0), (1, 1), (2, 2)]:
            self.assertLess(table.loc[frame, f"reference_{reference_id:04d}_rmsd_A"], .002)

    def test_amber_protonation_names_are_protein_and_preserved(self):
        source = self.simulations / "model_0/prot_meld_input.pdb"
        path = self.area / "amber.pdb"
        path.write_text(source.read_text().replace("ALA", "HIE"))
        _, _, keys, chemistry = backbone(path)
        self.assertEqual({key[1] for key in keys}, {"HIS"})
        self.assertEqual(set(chemistry), {"HIE"})

    def test_three_models_frame_provenance_and_report(self):
        result = self.invoke("--discard-frames", "1", "--stride", "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.output / "analysis_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["dynamics_launched"])
        self.assertEqual(len(manifest["runs"]), 3)
        frames = pd.read_csv(self.output / "reference_0000/frame_rmsd.csv")
        self.assertEqual(frames.replica_frame.tolist(), [1, 3, 5, 1, 3])
        self.assertEqual(frames.replica.tolist(), [0, 0, 0, 1, 1])
        counts = pd.read_csv(self.output / "assignment_populations.csv")
        summed = counts.groupby(["start_model", "scope", "cutoff_A", "discard_fraction_per_replica"])
        np.testing.assert_array_equal(summed["count"].sum(), summed.n_frames.first())
        np.testing.assert_allclose(summed.pct.sum(), 100)
        document = (self.output / "report.html").read_text()
        self.assertIn("data:image/png;base64,", document)
        self.assertNotIn('src="http', document)
        self.assertNotIn("src='http", document)
        self.assertIn("first model", document)
        # Existing output is rejected and never overwritten or marked failed.
        old = (self.output / "analysis_manifest.json").read_bytes()
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertEqual((self.output / "analysis_manifest.json").read_bytes(), old)

    def test_preflight_no_output_and_missing_replica_fails(self):
        source = self.simulations / "model_0/0.dcd"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        (self.simulations / "model_1/1.dcd").unlink()
        self.assertNotEqual(self.invoke("--dry-run").returncode, 0)
        self.assertFalse(self.output.exists())

    def test_protocol_mismatch_and_path_escape_rejected(self):
        path = self.simulations / "model_1/input.json"
        config = json.loads(path.read_text()); config["temperature_min_K"] = 310
        path.write_text(json.dumps(config))
        result = self.invoke("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("protocols differ", result.stderr)
        self.assertFalse(self.output.exists())
        manifest_path = self.simulations / "pipeline_manifest.json"
        manifest = json.loads(manifest_path.read_text()); manifest["runs"][0]["directory"] = ".."
        manifest_path.write_text(json.dumps(manifest))
        self.assertNotEqual(self.invoke("--dry-run").returncode, 0)

    def test_single_model_secondary_only_and_prefix_empty(self):
        path = self.simulations / "pipeline_manifest.json"
        manifest = json.loads(path.read_text()); manifest["runs"] = manifest["runs"][:1]
        path.write_text(json.dumps(manifest))
        result = self.invoke("--replicas", "0")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        table = pd.read_csv(self.output / "reference_0000/frame_rmsd.csv")
        self.assertIn(0, table.replica_frame.tolist())
        self.assertNotIn("ambiguous", table.assignment.tolist())
        result = self.invoke("--output", str(self.area / "empty"), "--discard-frames", "999")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.area / "empty").exists())

    def test_aligned_frame_export_uses_source_topology(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        export = self.area / "matches.dcd"
        result = subprocess.run([sys.executable, "-m", "analysis.export_frames", "--analysis", str(self.output),
                                 "--start-model", self.names[0], "--reference-model", self.names[1],
                                 "--cutoff", ".05", "--output", str(export)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        trajectory = md.load(str(export), top=str(self.simulations / "model_0/prot_meld_input.pdb"))
        ref = md.load(str(self.simulations / "model_1/prot_meld_input.pdb"))
        self.assertEqual(trajectory.n_frames, 3)
        self.assertEqual(trajectory.n_atoms, ref.n_atoms)
        self.assertLess(float(np.max(md.rmsd(trajectory, ref)*10)), .01)
        self.assertEqual(len(pd.read_csv(export.with_suffix(".frames.csv"))), 3)
        audit = subprocess.run([sys.executable, "-m", "analysis.validate_results", "--analysis", str(self.output),
                                "--full-rmsd"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(audit.returncode, 0, audit.stdout+audit.stderr)
        self.assertEqual(json.loads((self.output / "validation.json").read_text())["independent_comparisons"], 90)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional clustering requires the analysis environment")
    def test_bounded_clustering_sample_and_representatives(self):
        result = self.invoke("--cluster", "--cluster-max-frames", "24", "--cluster-min-size", "3",
                             "--cluster-min-samples", "2", "--stride", "2", "--cluster-discard-fraction", ".2")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        sample = pd.read_csv(self.output / "cluster_sample_assignments.csv")
        self.assertLessEqual(len(sample), 24)
        self.assertEqual(set(sample.replica_frame), {0, 2, 4})
        self.assertNotIn(0, set(sample[sample.replica == 0].replica_frame))
        populations = pd.read_csv(self.output / "cluster_sample_populations.csv")
        np.testing.assert_allclose(populations.groupby("start_model").sample_pct.sum(), 100)
        metadata = json.loads((self.output / "clustering_metadata.json").read_text())
        self.assertFalse(metadata["full_trajectory_assignments"])
        for path in (self.output / "cluster_representatives").glob("*.pdb"):
            self.assertEqual(md.load(str(path)).n_atoms, 40)


if __name__ == "__main__":
    unittest.main()
