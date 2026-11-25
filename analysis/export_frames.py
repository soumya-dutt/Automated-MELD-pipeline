"""Export aligned cutoff matches from one analyzed start, retaining its full topology.

Run from the project root: python -m analysis.export_frames --help
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import mdtraj as md
import numpy as np
import pandas as pd

from .metrics import backbone, digest


def align(xyz, indices, reference):
    all_xyz = np.asarray(xyz, dtype=np.float64)
    centers = all_xyz[:, indices].mean(axis=1)
    target = np.asarray(reference, dtype=np.float64)
    center = target.mean(axis=0)
    target = target-center
    u, _, vt = np.linalg.svd(np.einsum("fni,nj->fij", all_xyz[:, indices]-centers[:, None, :], target))
    signs = np.ones((len(xyz), 3))
    signs[:, -1] = np.where(np.linalg.det(u @ vt) < 0, -1, 1)
    return np.einsum("fni,fij->fnj", all_xyz-centers[:, None, :], (u*signs[:, None, :]) @ vt) + center


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--start-model", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--cutoff", type=float, default=4.)
    parser.add_argument("--scope", choices=["pooled", "replica_0"], default="pooled")
    parser.add_argument("--simulations", type=Path, help="Relocated simulation root, if originals have moved")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.cutoff) or args.cutoff <= 0:
        parser.error("cutoff must be positive and finite")
    temporary = None
    try:
        root = args.analysis.resolve()
        manifest = json.loads((root / "analysis_manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("Only a completed analysis can be exported.")
        mapping = {r["id"]: r for r in manifest["runs"]}
        run, ref = mapping[args.start_model], mapping[args.reference_model]

        def run_dir(record):
            if args.simulations is None:
                return Path(record["directory"])
            base = args.simulations.resolve()
            path = (base / record["relative_directory"]).resolve()
            if not path.is_relative_to(base):
                raise ValueError("Invalid relocated model directory.")
            return path

        own_dir, ref_dir = run_dir(run), run_dir(ref)
        for record, folder in [(run, own_dir), (ref, ref_dir)]:
            expected = next(s["sha256"] for s in record["sources"] if Path(s["path"]).name == "prot_meld_input.pdb")
            if digest(folder / "prot_meld_input.pdb") != expected:
                raise ValueError(f"Topology/reference changed since analysis: {folder}")
        own, indices, keys, _ = backbone(own_dir / "prot_meld_input.pdb")
        reference, ref_indices, ref_keys, _ = backbone(ref_dir / "prot_meld_input.pdb")
        if keys != ref_keys:
            raise ValueError("Backbone correspondence differs.")
        table = pd.read_csv(root / run["reference_id"] / "frame_rmsd.csv", dtype={"start_model": str})
        selected = table[table[f"{ref['reference_id']}_rmsd_A"] <= args.cutoff].copy()
        if args.scope == "replica_0":
            selected = selected[selected.replica == 0]
        output = args.output.resolve()
        provenance = output.with_suffix(".frames.csv")
        if output.suffix.lower() != ".dcd" or output.exists() or provenance.exists():
            raise ValueError("Use a fresh .dcd output path with no existing .frames.csv companion.")
        if any(output.is_relative_to(run_dir(r)) for r in manifest["runs"]):
            raise ValueError("Write exports outside simulation model directories.")
        if selected.empty:
            print("No matching frames; no DCD written.")
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix=".export-", suffix=".dcd", dir=output.parent, delete=False)
        temporary = Path(handle.name)
        handle.close()
        written = 0
        with md.formats.DCDTrajectoryFile(str(temporary), "w") as writer:
            for replica, sub in selected.groupby("replica", sort=True):
                item = next(f for f in run["files"] if f["replica"] == replica)
                path = own_dir / f"{replica}.dcd"
                saved = item["source"]
                stat = path.stat()
                if stat.st_size != saved["bytes"] or ("sha256" in saved and digest(path) != saved["sha256"]):
                    raise ValueError(f"DCD changed since analysis: {path}")
                if "sha256" not in saved and stat.st_mtime_ns != saved["mtime_ns"]:
                    raise ValueError(f"DCD mtime changed: {path}. Reanalyze, or use --hash-dcd in the original analysis for portable hashes.")
                selected_frames = set(sub.replica_frame.astype(int))
                offset = 0
                for chunk in md.iterload(str(path), top=own.top, chunk=500):
                    picks = [i for i in range(chunk.n_frames) if offset+i in selected_frames]
                    offset += chunk.n_frames
                    if picks:
                        coordinates = align(chunk.xyz[picks], indices, reference.xyz[0, ref_indices])
                        writer.write((coordinates * 10).astype(np.float32))  # DCD low-level API uses Å
                        written += len(picks)
        if written != len(selected):
            raise ValueError("Selected frame count does not match exported coordinates.")
        selected.to_csv(provenance, index=False)
        temporary.replace(output)
        temporary = None
        print(f"Exported {written} frames to {output}; topology: {own_dir / 'prot_meld_input.pdb'}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
