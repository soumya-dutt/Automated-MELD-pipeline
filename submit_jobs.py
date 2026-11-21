#!/usr/bin/env python3
"""Copy optional Slurm templates and preview submissions; use --submit to send jobs."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "templates"))
from structure_utils import manifest_runs, sha256, utc_now, write_json


def copy_job(template, folder, model_id):
    contents = template.read_text().replace("__JOB_NAME__", "meld_" + model_id[:64])
    target = folder / "job.slurm"
    if target.exists() and target.read_text() != contents:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup = folder / f"job.slurm.backup_{stamp}"
        shutil.copy2(target, backup)
        print(f"Backed up {target} to {backup.name}")
    if not target.exists() or target.read_text() != contents:
        target.write_text(contents)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=Path, default=ROOT / "simulations")
    parser.add_argument("--template", type=Path,
                        help="Copy this edited template into every run, backing up changed job files.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only (the default).")
    mode.add_argument("--submit", action="store_true", help="Actually call sbatch after all preflight checks.")
    parser.add_argument("--resubmit", action="store_true",
                        help="Include runs already recorded as submitted. Use only after checking previous jobs.")
    args = parser.parse_args()
    output = args.simulations.resolve()
    root, manifest = manifest_runs(output / "pipeline_manifest.json")
    if manifest["status"] != "complete":
        raise RuntimeError("Preparation is incomplete; finish pipeline.py before submission.")
    if args.template and not args.template.is_file():
        raise FileNotFoundError(args.template)
    history = root / "submissions.jsonl"
    with (root / ".submission.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another submission process is already using this output.")
        previous = []
        if history.exists():
            previous = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        submitted = {entry["model"] for entry in previous
                     if entry["status"] in ("submitted", "submission_result_uncertain")}
        jobs, problems = [], []
        # Validate every candidate before submitting the first job.
        for run in manifest["runs"]:
            folder = root / run["directory"]
            if run["id"] in submitted and not args.resubmit:
                print(f"SKIP already submitted: {run['id']}")
                continue
            for name, expected in run["staged_sha256"].items():
                if not (folder / name).is_file() or sha256(folder / name) != expected:
                    problems.append(f"{run['id']}: staged input changed/missing: {name}; prepare a new condition.")
            target = folder / "job.slurm"
            if args.template:
                target = copy_job(args.template.resolve(), folder, run["id"])
            elif not target.exists():
                target = copy_job(ROOT / "templates" / "job.slurm", folder, run["id"])
            text = target.read_text()
            tokens = sorted(set(re.findall(r"\bYOUR_[A-Z0-9_]+\b|__JOB_NAME__", text)))
            if tokens:
                problems.append(f"{run['id']}: edit Slurm placeholders: {', '.join(tokens)}")
            if "#SBATCH" not in text:
                problems.append(f"{run['id']}: job.slurm has no Slurm directives.")
            syntax = subprocess.run(["bash", "-n", str(target)], capture_output=True, text=True)
            if syntax.returncode:
                problems.append(f"{run['id']}: Bash syntax error: {syntax.stderr.strip()}")
            jobs.append((run, folder, target))
            print(f"{'SUBMIT' if args.submit else 'PREVIEW'}: {run['id']}: sbatch --parsable job.slurm")
        if problems:
            print("\nPreflight findings:\n" + "\n".join(problems), file=sys.stderr)
            if args.submit:
                raise RuntimeError("No jobs submitted: fix all preflight findings first.")
        if not args.submit:
            print(f"\nDry run: {len(jobs)} jobs considered; zero jobs submitted. Add --submit when ready.")
            return
        executable = shutil.which("sbatch")
        if jobs and executable is None:
            raise RuntimeError("sbatch is not available. Run submission on your Slurm HPC.")
        for run, folder, target in jobs:
            result = subprocess.run([executable, "--parsable", target.name], cwd=folder,
                                    capture_output=True, text=True)
            entry = {"time_utc": utc_now(), "model": run["id"], "directory": run["directory"],
                     "script_sha256": sha256(target), "returncode": result.returncode,
                     "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
            answer = result.stdout.strip()
            if result.returncode == 0 and re.fullmatch(r"[0-9]+(?:;[^\s;]+)?", answer):
                entry.update(status="submitted", job_id=answer.split(";")[0])
            else:
                entry["status"] = "failed" if result.returncode else "submission_result_uncertain"
            with history.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
                handle.flush()
            manifest["last_submission_attempt_utc"] = entry["time_utc"]
            if entry["status"] == "submitted":
                manifest["jobs_submitted"] = True
            elif entry["status"] == "submission_result_uncertain":
                manifest["submission_result_uncertain"] = True
            write_json(root / "pipeline_manifest.json", manifest)
            if entry["status"] != "submitted":
                raise RuntimeError(f"Submission stopped for {run['id']}; inspect {history} before retrying.")
            print(f"Submitted {run['id']}: job {entry['job_id']}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
