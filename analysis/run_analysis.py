"""Manifest-driven backbone alignment, merged DCDs and structural ensemble analysis."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "meld-analysis-matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "meld-analysis-numba"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mdtraj as md
import numpy as np
import pandas as pd

from .metrics import assignments, backbone, digest, dump_json, matched_backbone, rmsd_A, signature
from .trajectories import aligned_chunks, export_reference_structures

PROJECT = Path(__file__).resolve().parents[1]


def child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path must be inside {root}: {relative}")
    return path


def discover(root, replicas=None, hash_dcd=False):
    """Read and validate all inputs before creating any analysis output."""
    manifest_path = root / "pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or not manifest.get("runs"):
        raise ValueError("A complete, nonempty pipeline_manifest.json is required.")
    runs, keys0, chemistry0, physics0, contact0 = [], None, None, None, None
    seen_ids, seen_dirs = set(), set()
    for record in manifest["runs"]:
        identifier = record["id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            raise ValueError("Manifest model IDs must be unique nonempty strings.")
        seen_ids.add(identifier)
        folder = child(root, record["directory"])
        if folder in seen_dirs:
            raise ValueError(f"Repeated model directory: {folder}")
        seen_dirs.add(folder)
        pdb = folder / "prot_meld_input.pdb"
        config_path = folder / "input.json"
        config = json.loads(config_path.read_text())
        count = config["n_replicas"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Invalid n_replicas: {folder}")
        selected = list(range(count)) if replicas is None else sorted(set(replicas))
        if not selected or min(selected) < 0 or max(selected) >= count:
            raise ValueError(f"Requested replicas must be in 0..{count-1}: {folder}")
        ref, indices, keys, chemistry = backbone(pdb)
        if keys0 is not None and keys != keys0:
            raise ValueError(f"Backbone sequence/correspondence differs: {pdb}")
        if chemistry0 is not None and chemistry != chemistry0:
            raise ValueError(f"Prepared residue chemistry differs: {pdb}. Analyze separate conditions.")
        keys0, chemistry0 = keys, chemistry
        physics = {k: v for k, v in config.items()
                   if k not in ("run_id", "source_structure", "langevin_seed")}
        if physics0 is not None and physics != physics0:
            raise ValueError(f"MELD protocols differ across starts: {config_path}")
        physics0 = physics
        sources = [signature(pdb, True), signature(config_path, True)]
        contacts = {}
        for label, relative in config.get("contact_files", {}).items():
            path = child(folder, relative)
            contacts[label] = digest(path)
            expected = record.get("staged_sha256", {}).get(relative)
            if expected and contacts[label] != expected:
                raise ValueError(f"Prepared contacts changed since staging: {path}")
            sources.append(signature(path, True))
        if contact0 is not None and contacts != contact0:
            raise ValueError(f"Shared restraint files differ: {folder}")
        contact0 = contacts
        for name in ("prot_meld_input.pdb", "input.json"):
            expected = record.get("staged_sha256", {}).get(name)
            if name == "prot_meld_input.pdb":
                expected = expected or record.get("prepared_sha256")
            if expected and digest(folder / name) != expected:
                raise ValueError(f"Prepared input changed since staging: {folder / name}")
        saved_path = folder / "setup_report.json"
        if saved_path.exists():
            saved = json.loads(saved_path.read_text())
            if saved.get("effective_config") != config:
                raise ValueError(f"input.json differs from saved setup: {folder}")
            for relative in ["prot_meld_input.pdb", "input.json", *config.get("contact_files", {}).values()]:
                expected = saved.get("input_sha256", {}).get(relative)
                if expected and digest(child(folder, relative)) != expected:
                    raise ValueError(f"Scientific input changed since setup: {folder / relative}")
            sources.append(signature(saved_path, True))
        files = []
        for replica in selected:
            path = folder / f"{replica}.dcd"
            with md.formats.DCDTrajectoryFile(str(path)) as handle:
                frames = len(handle)
                first = handle.read(n_frames=1)[0]
            if frames < 1 or first.shape[1] != ref.n_atoms:
                raise ValueError(f"Empty trajectory or topology atom mismatch: {path}")
            files.append({"replica": replica, "frames": frames, "source": signature(path, hash_dcd)})
        extras = [p.name for p in folder.glob("*.dcd")
                  if re.fullmatch(r"\d+", p.stem) and int(p.stem) >= count]
        if extras:
            raise ValueError(f"Unexpected replica files beyond configured range in {folder}: {extras}")
        runs.append({"id": identifier, "directory": str(folder), "relative_directory": record["directory"], "pdb": str(pdb),
                     "reference_id": f"reference_{len(runs):04d}", "config": config,
                     "files": files, "sources": sources, "reference": ref, "indices": indices,
                     "backbone_reference": matched_backbone(ref, indices)})
    return manifest_path, runs


def select_sample_rows(run, args, limit):
    """Deterministic balanced-by-start subset; RMSD analysis still uses all retained frames."""
    eligible = []
    count = 0
    for item in run["files"]:
        if args.cluster_scope == "replica_0" and item["replica"] != 0:
            continue
        start = max(args.discard_frames, int(np.floor(item["frames"] * args.cluster_discard_fraction)))
        start += (args.discard_frames-start) % args.stride
        size = max(0, (item["frames"]-start+args.stride-1)//args.stride)
        eligible.append((item["replica"], start, size, count))
        count += size
    if not count:
        raise ValueError(f"No frames for clustering after exclusions: {run['id']}")
    chosen = np.linspace(0, count-1, count if limit is None else min(limit, count), dtype=int)
    return {(replica, int(start+(index-offset)*args.stride))
            for replica, start, size, offset in eligible
            for index in chosen[(chosen >= offset) & (chosen < offset+size)]}


def analyze_run(run, runs, output, args, columns, sample_limit):
    folder = output / run["reference_id"]
    folder.mkdir()
    selected = select_sample_rows(run, args, sample_limit) if args.cluster else set()
    pieces, sampled_xyz, sampled_rows = [], [], []
    audit_max, audit_n = 0., 0
    for item, frames, total_frames, chunk in aligned_chunks(run, folder, args.chunk_size):
        path = item["source"]["path"]
        if not np.isfinite(chunk.xyz).all():
            raise ValueError(f"Non-finite trajectory coordinates: {path}")
        keep = (frames >= args.discard_frames) & ((frames-args.discard_frames) % args.stride == 0)
        if not np.any(keep):
            continue
        trajectory = chunk[np.flatnonzero(keep)]
        frames = frames[keep]
        total_frames = total_frames[keep]
        values = []
        for other in runs:
            target = other["reference"].xyz[0, other["indices"]]
            values.append(rmsd_A(trajectory.xyz, target))
            # Independent QCP check on the first retained frames in each chunk.
            test = trajectory[:min(3, trajectory.n_frames)]
            reference = other["backbone_reference"]
            comparison = md.rmsd(test, reference, parallel=False) * 10
            if not np.isfinite(comparison).all():
                raise ValueError(f"Independent RMSD audit produced non-finite values: {path}")
            error = float(np.max(np.abs(values[-1][:len(test)] - comparison)))
            audit_max = max(audit_max, error)
            audit_n += len(test)
            if error > .025:
                raise ValueError(f"Independent RMSD audit differs by {error:.5f} A: {path}")
        distances = np.column_stack(values)
        hits, labels = assignments(distances, args.main_cutoff)
        table = pd.DataFrame(distances, columns=columns)
        table.insert(0, "replica_frame", frames)
        table.insert(0, "replica", item["replica"])
        table.insert(0, "total_frame", total_frames)
        table.insert(0, "start_model", run["id"])
        table["assignment"] = labels
        table["n_references_within_cutoff"] = hits.sum(axis=1)
        pieces.append(table)
        for index, frame in enumerate(frames):
            if (item["replica"], int(frame)) in selected:
                sampled_xyz.append(trajectory.xyz[index].copy())
                sampled_rows.append({"start_model": run["id"], "replica": item["replica"],
                                     "replica_frame": int(frame), "total_frame": int(total_frames[index]),
                                     "reference_id": run["reference_id"]})
    if not pieces:
        raise ValueError(f"No frames retained: {run['id']}")
    frame_table = pd.concat(pieces, ignore_index=True)
    frame_table.to_csv(folder / "frame_rmsd.csv", index=False)
    export_reference_structures(run, runs, folder, frame_table, args.main_cutoff, not args.no_extract)
    return frame_table, sampled_xyz, sampled_rows, {"start_model": run["id"],
           "independent_comparisons": audit_n, "max_QCP_difference_A": audit_max}


def populations(run, table, refs, columns, args):
    rows, partitions = [], []
    lengths = {item["replica"]: item["frames"] for item in run["files"]}
    scopes = [("pooled", table)] + [(f"replica_{r}", sub) for r, sub in table.groupby("replica")]
    for scope, original in scopes:
        for discard in args.discard_fractions:
            minimum = original.replica.map(lambda r: max(args.discard_frames, int(np.floor(lengths[r]*discard))))
            sub = original[original.replica_frame >= minimum]
            for cutoff in args.cutoffs:
                hits, labels = assignments(sub[columns].to_numpy(), cutoff)
                for index, ref in enumerate(refs):
                    rows.append({"start_model": run["id"], "scope": scope, "cutoff_A": cutoff,
                                 "discard_fraction_per_replica": discard, "n_frames": len(sub),
                                 "reference_id": ref["reference_id"], "reference_model": ref["id"],
                                 "within_count": int(hits[:, index].sum()),
                                 "within_pct": 100*float(hits[:, index].mean()) if len(sub) else None,
                                 "exclusive_count": int(np.sum(labels == ref["reference_id"]))})
                for label in [r["reference_id"] for r in refs] + ["ambiguous", "unassigned"]:
                    count = int(np.sum(labels == label))
                    partitions.append({"start_model": run["id"], "scope": scope, "cutoff_A": cutoff,
                                       "discard_fraction_per_replica": discard, "n_frames": len(sub),
                                       "assignment": label, "count": count,
                                       "pct": 100*count/len(sub) if len(sub) else None})
    return rows, partitions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=Path, default=PROJECT / "simulations")
    parser.add_argument("--output", type=Path, default=PROJECT / "analysis_results")
    parser.add_argument("--dry-run", "--preflight-only", action="store_true", dest="dry_run")
    parser.add_argument("--replicas", type=int, nargs="+", help="Default: all configured indices; e.g. --replicas 0")
    parser.add_argument("--main-cutoff", type=float, default=4.)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[2., 3., 4., 5.])
    parser.add_argument("--discard-frames", type=int, default=0, help="Absolute prefix to omit in EVERY numbered DCD")
    parser.add_argument("--discard-fractions", type=float, nargs="+", default=[0., .1, .2])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-dcd", action="store_true", help="Full SHA256 of trajectories in addition to size/mtime")
    clustering = parser.add_mutually_exclusive_group()
    clustering.add_argument("--cluster", dest="cluster", action="store_true", help="Run structural clustering (default)")
    clustering.add_argument("--no-cluster", dest="cluster", action="store_false", help="Skip structural clustering")
    parser.set_defaults(cluster=True)
    parser.add_argument("--cluster-scope", choices=["pooled", "replica_0"], default="pooled")
    parser.add_argument("--cluster-discard-fraction", type=float, default=0.)
    parser.add_argument("--cluster-max-frames", type=int, default=0, help="Total sample budget; 0 uses every eligible frame (default)")
    parser.add_argument("--cluster-fraction", type=float, default=.01)
    parser.add_argument("--cluster-min-size", type=int, default=40)
    parser.add_argument("--cluster-min-samples", type=int, default=10)
    embedding = parser.add_mutually_exclusive_group()
    embedding.add_argument("--umap", dest="umap", action="store_true", help="UMAP visualization (default with clustering)")
    embedding.add_argument("--no-umap", dest="umap", action="store_false", help="Use PCA visualization without UMAP")
    parser.set_defaults(umap=None)
    parser.add_argument("--no-extract", action="store_true", help="Skip cutoff-selected DCDs; aligned total.dcd and closest PDBs are still saved")
    args = parser.parse_args(argv)
    numbers = [args.main_cutoff, *args.cutoffs, *args.discard_fractions,
               args.cluster_discard_fraction, args.cluster_fraction]
    if not all(np.isfinite(numbers)):
        parser.error("Numeric settings must be finite.")
    if min(args.main_cutoff, *args.cutoffs) <= 0 or min(args.stride, args.chunk_size, args.threads) < 1:
        parser.error("Cutoffs, stride, chunk size and threads must be positive.")
    if args.discard_frames < 0 or any(not 0 <= x < 1 for x in args.discard_fractions + [args.cluster_discard_fraction]):
        parser.error("Discard frames must be nonnegative; fractions must be in [0,1).")
    if not 0 < args.cluster_fraction <= 1 or min(args.cluster_min_size, args.cluster_min_samples) < 2:
        parser.error("Invalid clustering size/fraction.")
    args.umap = args.cluster if args.umap is None else args.umap
    if args.cluster_max_frames < 0:
        parser.error("--cluster-max-frames must be nonnegative; 0 means all eligible frames.")
    if args.umap and not args.cluster:
        parser.error("--umap requires --cluster.")
    args.cutoffs = sorted(set(args.cutoffs + [args.main_cutoff]))
    args.discard_fractions = sorted(set(args.discard_fractions + [0.]))
    output = args.output.resolve()
    manifest = None
    try:
        root = args.simulations.resolve()
        preparation_path, runs = discover(root, args.replicas, args.hash_dcd)
        if args.cluster and 0 < args.cluster_max_frames < 2*len(runs):
            raise ValueError("--cluster-max-frames must allow at least two frames per starting model.")
        if output == root or any(output == Path(r["directory"]) or output.is_relative_to(Path(r["directory"]))
                                 or Path(r["directory"]).is_relative_to(output) for r in runs):
            raise ValueError("Analysis output must be separate from simulation model directories.")
        sample_limit = None if args.cluster_max_frames == 0 else args.cluster_max_frames // len(runs)
        for run in runs:
            if not any(item["frames"] > args.discard_frames for item in run["files"]):
                raise ValueError(f"No frames after --discard-frames: {run['id']}")
            if args.cluster:
                select_sample_rows(run, args, sample_limit)
        packages = ["numpy", "mdtraj", "pandas", "matplotlib", "threadpoolctl"]
        if args.cluster:
            from sklearn.cluster import HDBSCAN  # fail before outputs if dependency is missing
            packages.append("scikit-learn")
        if args.umap:
            packages.append("umap-learn")
        software = {name: version(name) for name in packages}
        sources = [signature(preparation_path, True)] + [s for r in runs for s in r["sources"]]
        sources += [f["source"] for r in runs for f in r["files"]]
        serial_runs = [{k: v for k, v in r.items() if k not in ("reference", "indices", "backbone_reference")} for r in runs]
        settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
        manifest = {"schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "settings": settings, "software": software, "python": sys.version,
                    "runs": serial_runs, "sources": sources,
                    "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                    "primary_scope": "pooled", "secondary_scope": "replica_0",
                    "numbered_DCD_frame_zero_is_sample": True, "total_dcd_used": True,
                    "alignment": {"atoms": ["N", "CA", "C", "O"], "reference": "each start's prepared input PDB",
                                  "full_atom_total_dcd": "reference_*/total.dcd", "synthetic_initial_frame": False,
                                  "all_selected_replica_frames_included": True},
                    "dynamics_launched": False, "equilibrium_populations_established": False}
        print(f"Validated {len(runs)} starting models, {sum(len(r['files']) for r in runs)} DCD files, "
              f"{sum(f['frames'] for r in runs for f in r['files']):,} input frames.", flush=True)
        if args.dry_run:
            print("Preflight only: no analysis output written; headers checked, trajectories not fully scanned.")
            return 0
        if output.exists():
            raise ValueError(f"Output already exists. Choose a fresh --output directory: {output}")
        output.mkdir(parents=True)
        dump_json(output / "analysis_manifest.json", manifest)
        columns = [r["reference_id"] + "_rmsd_A" for r in runs]
        rows, partitions, audits, sample_xyz, sample_rows = [], [], [], [], []
        reference_rows = []
        for a in runs:
            for b in runs:
                reference_rows.append({"model_a": a["id"], "model_b": b["id"],
                                       "reference_a": a["reference_id"], "reference_b": b["reference_id"],
                                       "rmsd_A": float(rmsd_A(a["reference"].xyz[:, a["indices"]],
                                                               b["reference"].xyz[0, b["indices"]])[0])})
        pd.DataFrame(reference_rows).to_csv(output / "reference_rmsd.csv", index=False)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            for run in runs:
                print(f"Analyzing {run['id']}", flush=True)
                table, xyz, provenance, audit = analyze_run(run, runs, output, args, columns, sample_limit)
                summary, partition = populations(run, table, runs, columns, args)
                rows.extend(summary)
                partitions.extend(partition)
                audits.append(audit)
                sample_xyz.extend(xyz)
                sample_rows.extend(provenance)
            if args.cluster:
                from .cluster import cluster_sample
                cluster_sample(sample_xyz, sample_rows, runs, output, args)
        all_pop = pd.DataFrame(rows)
        all_pop.to_csv(output / "reference_populations.csv", index=False)
        pd.DataFrame(partitions).to_csv(output / "assignment_populations.csv", index=False)
        for scope, filename in [("pooled", "primary_pooled_summary.csv"), ("replica_0", "secondary_replica0_summary.csv")]:
            all_pop[(all_pop.scope == scope) & (all_pop.cutoff_A == args.main_cutoff)
                    & (all_pop.discard_fraction_per_replica == 0)].to_csv(output / filename, index=False)
        # Prevent labeling partial/changing inputs as a completed analysis.
        if any(signature(s["path"], "sha256" in s) != s for s in sources):
            raise ValueError("An input changed during analysis. Use completed, unchanged extracted trajectories.")
        dump_json(output / "numerical_audit.json", {"status": "passed", "scope": "sampled QCP cross-check; all reported RMSDs use Kabsch",
                                                   "runs": audits})
        from .report import build_report
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        build_report(output, manifest)
        manifest["status"] = "complete"
        dump_json(output / "analysis_manifest.json", manifest)
        print(f"Complete: {output / 'report.html'}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, np.linalg.LinAlgError) as exc:
        if manifest is not None and output.exists() and (output / "analysis_manifest.json").exists():
            # Only mark outputs created by THIS invocation.
            current = json.loads((output / "analysis_manifest.json").read_text())
            if current.get("created_utc") == manifest["created_utc"]:
                manifest.update(status="failed", error=str(exc))
                dump_json(output / "analysis_manifest.json", manifest)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
