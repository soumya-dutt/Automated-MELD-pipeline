#!/usr/bin/env python3
"""Run the single-simulation replica extraction loop in every prepared model folder."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=Path, default=ROOT / "simulations")
    parser.add_argument("--first-replica", type=int, default=0)
    parser.add_argument("--last-replica", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.first_replica < 0 or (args.last_replica is not None and args.last_replica < args.first_replica):
        parser.error("Replica bounds must be nonnegative and last >= first.")
    root = args.simulations.resolve()
    manifest = json.loads((root / "pipeline_manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest["runs"]:
        raise RuntimeError("A complete, nonempty preparation manifest is required.")
    folders = []
    for run in manifest["runs"]:
        folder = (root / run["directory"]).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise RuntimeError(f"Missing/invalid model folder: {run['directory']}")
        script = folder / "extract_trajectories.sh"
        if not script.exists():
            script = ROOT / "templates" / "extract_trajectories.sh"
        if not script.is_file():
            raise FileNotFoundError(script)
        folders.append((run, folder, script))
    results = []
    for index, (run, folder, script) in enumerate(folders, 1):
        print(f"\n[{index}/{len(folders)}] Extracting {run['id']}", flush=True)
        command = ["bash", str(script), "--run-dir", str(folder), "--first-replica", str(args.first_replica)]
        if args.last_replica is not None:
            command += ["--last-replica", str(args.last_replica)]
        if args.overwrite:
            command.append("--overwrite")
        if args.dry_run:
            command.append("--dry-run")
        result = subprocess.run(command, cwd=folder)
        results.append({"model": run["id"], "directory": run["directory"], "returncode": result.returncode})
    summary = {"time_utc": datetime.now(timezone.utc).isoformat(), "dry_run": args.dry_run,
               "first_replica": args.first_replica, "last_replica_override": args.last_replica,
               "overwrite": args.overwrite, "results": results,
               "all_passed": all(row["returncode"] == 0 for row in results)}
    if not args.dry_run:
        (root / "extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    failed = sum(row["returncode"] != 0 for row in results)
    print(f"\n{'Preview' if args.dry_run else 'Extraction'} complete: {len(results)-failed} folders OK; {failed} failed.")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
