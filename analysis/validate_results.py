"""Audit portable result accounting, optionally recomputing every RMSD with MDTraj.

From repository root: python -m analysis.validate_results --analysis analysis_results
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mdtraj as md
import numpy as np
import pandas as pd

from .metrics import assignments, backbone, digest, dump_json, matched_backbone
from .run_analysis import child, populations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--full-rmsd", action="store_true", help="Reread all retained coordinates; can be expensive")
    parser.add_argument("--simulations", type=Path, help="Optional relocated simulation root")
    args = parser.parse_args(argv)
    try:
        root = args.analysis.resolve()
        manifest = json.loads((root / "analysis_manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("A complete analysis manifest is required.")
        settings = SimpleNamespace(**manifest["settings"])
        runs = manifest["runs"]
        columns = [r["reference_id"] + "_rmsd_A" for r in runs]
        tables, expected_rows, expected_partitions = {}, [], []
        for run in runs:
            table = pd.read_csv(root / run["reference_id"] / "frame_rmsd.csv", dtype={"start_model": str})
            expected = [(item["replica"], frame) for item in run["files"]
                        for frame in range(settings.discard_frames, item["frames"], settings.stride)]
            if list(zip(table.replica, table.replica_frame)) != expected or not (table.start_model == run["id"]).all():
                raise ValueError(f"Frame provenance differs: {run['id']}")
            if not np.isfinite(table[columns].to_numpy()).all():
                raise ValueError(f"Non-finite RMSDs: {run['id']}")
            hits, labels = assignments(table[columns].to_numpy(), settings.main_cutoff)
            if not np.array_equal(table.assignment, labels) or not np.array_equal(table.n_references_within_cutoff, hits.sum(axis=1)):
                raise ValueError(f"Frame assignments differ: {run['id']}")
            summary, partition = populations(run, table, runs, columns, settings)
            expected_rows.extend(summary); expected_partitions.extend(partition)
            tables[run["id"]] = table
        for filename, values in [("reference_populations.csv", expected_rows),
                                 ("assignment_populations.csv", expected_partitions)]:
            actual = pd.read_csv(root / filename, dtype={"start_model": str, "reference_model": str})
            expected = pd.DataFrame(values)
            # CSV roundtrip converts nullable numeric values to NaN.
            expected = expected.replace({None: np.nan})
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, atol=1e-8, rtol=1e-8)
        all_pop = pd.DataFrame(expected_rows)
        for scope, filename in [("pooled", "primary_pooled_summary.csv"), ("replica_0", "secondary_replica0_summary.csv")]:
            expected = all_pop[(all_pop.scope == scope) & (all_pop.cutoff_A == settings.main_cutoff)
                               & (all_pop.discard_fraction_per_replica == 0)].reset_index(drop=True)
            actual = pd.read_csv(root / filename, dtype={"start_model": str, "reference_model": str})
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, atol=1e-8, rtol=1e-8)
        comparisons, maximum = 0, 0.
        if args.full_rmsd:
            references, loaded = [], []
            for run in runs:
                folder = Path(run["directory"]) if args.simulations is None else child(args.simulations.resolve(), run["relative_directory"])
                pdb = folder / "prot_meld_input.pdb"
                expected = next(s["sha256"] for s in run["sources"] if Path(s["path"]).name == pdb.name)
                if digest(pdb) != expected:
                    raise ValueError(f"Reference changed: {pdb}")
                ref, indices, _, _ = backbone(pdb)
                references.append(matched_backbone(ref, indices))
                loaded.append((folder, pdb, indices))
            for run, (folder, pdb, indices), own in zip(runs, loaded, references):
                table = tables[run["id"]].set_index(["replica", "replica_frame"])
                for item in run["files"]:
                    path = folder / f"{item['replica']}.dcd"
                    saved = item["source"]; stat = path.stat()
                    if stat.st_size != saved["bytes"] or (digest(path) != saved["sha256"] if "sha256" in saved else stat.st_mtime_ns != saved["mtime_ns"]):
                        raise ValueError(f"Trajectory changed: {path}")
                    offset = 0
                    sorted_indices = np.sort(indices)
                    order = np.searchsorted(sorted_indices, indices)
                    for chunk in md.iterload(str(path), top=str(pdb), atom_indices=sorted_indices, chunk=settings.chunk_size):
                        frame = np.arange(offset, offset+chunk.n_frames); offset += chunk.n_frames
                        keep = (frame >= settings.discard_frames) & ((frame-settings.discard_frames) % settings.stride == 0)
                        if not keep.any():
                            continue
                        coordinates = md.Trajectory(chunk.xyz[keep][:, order], own.top)
                        keys = [(item["replica"], int(f)) for f in frame[keep]]
                        for reference, column in zip(references, columns):
                            measured = md.rmsd(coordinates, reference, parallel=False)*10
                            if not np.isfinite(measured).all():
                                raise ValueError(f"Non-finite independent RMSD: {path}")
                            expected = table.loc[keys, column].to_numpy()
                            difference = np.abs(measured-expected)
                            maximum = max(maximum, float(difference.max()))
                            comparisons += len(measured)
                            if difference.max() > .025:
                                raise ValueError(f"Independent RMSD discrepancy: {path}")
                    if offset != item["frames"]:
                        raise ValueError(f"DCD count differs: {path}")
        result = {"status": "passed", "start_models": len(runs),
                  "retained_frames": sum(len(t) for t in tables.values()),
                  "population_tables_recomputed": True, "full_independent_RMSD": args.full_rmsd,
                  "independent_comparisons": comparisons, "max_RMSD_difference_A": maximum if comparisons else None}
        dump_json(root / "validation.json", result)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, AssertionError) as exc:
        if args.analysis.is_dir():
            dump_json(args.analysis / "validation.json", {"status": "failed", "error": str(exc)})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
