#!/usr/bin/env python3
"""Prepare all AF2 PDB models of one protein for normal MELD. No MD or submission."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
sys.path.insert(0, str(TEMPLATES))
from structure_utils import load_pdb, require_same_sequence, sha256, utc_now, write_json


@contextmanager
def preparation_lock(output):
    with (output / ".preparation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another preparation process is already using this output.")
        yield


def validate_config(config):
    numbers = [config[key] for key in ("n_replicas", "max_cycles", "timesteps_per_cycle",
                                      "equilibration_cycles", "ramp_end_cycle")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in numbers):
        raise ValueError("Replica/cycle/step counts must be integers.")
    if config["n_replicas"] < 2 or config["timesteps_per_cycle"] < 1:
        raise ValueError("Need at least two replicas and positive MD steps per cycle.")
    if not 1 < config["ramp_end_cycle"] <= config["equilibration_cycles"] < config["max_cycles"]:
        raise ValueError("Need 1 < ramp end <= equilibration cycles < max cycles.")
    if not 0 < config["temperature_min_K"] <= config["temperature_max_K"]:
        raise ValueError("Invalid temperature range.")
    if config["pdb_file"] != "prot_meld_input.pdb" or config["contact_files"] != {
        "core": "contacts/core_contacts.csv", "alternatives": "contacts/alternative_contacts.csv"
    }:
        raise ValueError("Keep the standard PDB and contact filenames in the protocol template.")
    for label in ("core", "alternatives"):
        values = config["collections"][label]
        if not 0 <= values["confidence"] <= 1:
            raise ValueError(f"{label} confidence must be in [0, 1].")
        if not 0 <= values["alpha_min"] < values["alpha_max"] <= 1 or values["scaler_factor"] < 1:
            raise ValueError(f"Invalid {label} scaler settings.")
    setting = config["contact_generation"]
    if not 0 < setting["cutoff_A"] <= setting["alternative_upper_bound_A"]:
        raise ValueError("Need 0 < contact cutoff <= alternative upper bound.")
    if not isinstance(setting["minimum_sequence_separation"], int) or setting["minimum_sequence_separation"] < 1:
        raise ValueError("Minimum sequence separation must be a positive integer.")
    if not 0 <= setting["minimum_core_plddt"] <= 100:
        raise ValueError("Core pLDDT threshold must be between 0 and 100.")
    if setting["core_maximum_spread_A"] < 0 or setting["core_margin_A"] < 0 or setting["force_constant_kj_mol_nm2"] <= 0:
        raise ValueError("Invalid contact geometry or force constant.")
    def check_finite(value):
        if isinstance(value, dict):
            for item in value.values(): check_finite(item)
        elif isinstance(value, list):
            for item in value: check_finite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Configuration contains a non-finite number.")
    check_finite(config)


def run_logged(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        handle.write("Command: " + repr([str(value) for value in command]) + "\n\n")
        handle.flush()
        process = subprocess.run([str(value) for value in command], cwd=cwd,
                                 stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Command failed with status {process.returncode}; see {log_path}")


def check_complete(output, manifest):
    for run in manifest["runs"]:
        folder = output / run["directory"]
        for name, expected in run["staged_sha256"].items():
            if sha256(folder / name) != expected:
                raise RuntimeError(f"Staged input changed: {folder / name}. Use a new output for a changed protocol.")
    for name, expected in manifest["contact_sha256"].items():
        if sha256(output / name) != expected:
            raise RuntimeError(f"Shared contact file changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, default=ROOT / "structures")
    parser.add_argument("--output", type=Path, default=ROOT / "simulations")
    parser.add_argument("--config", type=Path, default=TEMPLATES / "input.json")
    parser.add_argument("--core-confidence", type=float)
    parser.add_argument("--alternative-confidence", type=float)
    parser.add_argument("--resume", action="store_true", help="Resume unchanged preparation or verify a complete output.")
    args = parser.parse_args()
    source, output = args.structures.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error(f"Missing structures directory: {source}")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Structures and output must be separate, non-nested directories.")
    paths = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".pdb"
                   and not any(part.startswith(".") for part in path.relative_to(source).parts))
    if not paths:
        parser.error(f"No PDB files were found under {source}")
    config = json.loads(args.config.read_text())
    for label, value in (("core", args.core_confidence), ("alternatives", args.alternative_confidence)):
        if value is not None:
            config["collections"][label]["confidence"] = value
    validate_config(config)
    raw_models = [load_pdb(path) for path in paths]
    for model in raw_models:
        require_same_sequence(raw_models[0], model)
        if any(name == "CYX" for name in model["residue_names"]):
            raise ValueError("CYX/disulfide topology requires explicit covalent-bond preparation before this standard protocol.")
    inputs = [{"relative_path": str(path.relative_to(source)), "sha256": sha256(path)} for path in paths]
    template_hashes = {str(path.relative_to(TEMPLATES)): sha256(path)
                       for path in sorted(TEMPLATES.iterdir()) if path.is_file()}
    if output.exists() and not args.resume:
        parser.error(f"Output exists: {output}. Use --resume for unchanged inputs, or choose a new --output.")
    if not output.exists():
        output.mkdir(parents=True)
    with preparation_lock(output):
        manifest_path = output / "pipeline_manifest.json"
        if args.resume:
            if not manifest_path.is_file():
                raise RuntimeError("No preparation manifest exists in this output; choose a new output directory.")
            manifest = json.loads(manifest_path.read_text())
            auxiliary_templates = {"job.slurm", "extract_trajectories.sh"}
            old_protocol_templates = {key: value for key, value in manifest["template_sha256"].items() if key not in auxiliary_templates}
            new_protocol_templates = {key: value for key, value in template_hashes.items() if key not in auxiliary_templates}
            if manifest["inputs"] != inputs or manifest["protocol"] != config or old_protocol_templates != new_protocol_templates:
                raise RuntimeError("Inputs, protocol, or templates changed. Use a new --output to preserve the existing condition.")
            if manifest["status"] == "complete":
                check_complete(output, manifest)
                print(f"Already complete: {len(manifest['runs'])} model folders verified in {output}")
                return
            if any((output / run["directory"] / name).exists() for run in manifest["runs"] for name in ("Data", "Logs")):
                raise RuntimeError("MELD Data/Logs exists in incomplete preparation. Use a new output; existing runs are protected.")
        else:
            runs = []
            for index, (path, entry) in enumerate(zip(paths, inputs), 1):
                slug = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")[:48] or "model"
                name = f"model_{index:04d}_{slug}"
                runs.append({"id": name, "directory": name, "source_relative": entry["relative_path"],
                             "source_sha256": entry["sha256"], "status": "pending"})
            manifest = {"schema_version": 1, "created_utc": utc_now(), "status": "preparing",
                        "source_directory_at_preparation": str(source), "inputs": inputs,
                        "protocol": config, "template_sha256": template_hashes, "runs": runs,
                        "dynamics_launched": False, "jobs_submitted": False}
            write_json(manifest_path, manifest)
        tools = {name: shutil.which(name) for name in ("pdb4amber", "tleap")}
        if not all(tools.values()):
            raise RuntimeError("Activate meld_new with AmberTools available: missing " +
                               ", ".join(name for name, value in tools.items() if value is None))
        manifest["preparation_tools"] = tools
        try:
            for index, (run, raw) in enumerate(zip(manifest["runs"], raw_models)):
                folder = output / run["directory"]
                folder.mkdir(exist_ok=True)
                if run["status"] == "prepared":
                    if sha256(folder / "prot_meld_input.pdb") != run["prepared_sha256"]:
                        raise RuntimeError(f"Prepared PDB changed: {folder}")
                    if sha256(folder / "raw_af2.pdb") != run["source_sha256"]:
                        raise RuntimeError(f"Copied raw PDB changed: {folder}")
                    print(f"[{index+1}/{len(paths)}] Reusing {run['id']}", flush=True)
                    continue
                print(f"[{index+1}/{len(paths)}] Preparing {run['id']}", flush=True)
                shutil.copy2(raw["path"], folder / "raw_af2.pdb")
                preparation = folder / "preparation"
                preparation.mkdir(exist_ok=True)
                shutil.copy2(TEMPLATES / "tleap.in", preparation / "tleap.in")
                for name in ("pdb4amber.pdb", "prot_meld_input.pdb"):
                    (preparation / name).unlink(missing_ok=True)
                run_logged([tools["pdb4amber"], "-i", "../raw_af2.pdb", "-o", "pdb4amber.pdb"],
                           preparation, folder / "preparation_logs" / "pdb4amber.log")
                run_logged([tools["tleap"], "-f", "tleap.in"], preparation,
                           folder / "preparation_logs" / "tleap.log")
                leap_log = (folder / "preparation_logs" / "tleap.log").read_text()
                if re.search(r"FATAL:", leap_log, re.I) or any(int(n) for n in re.findall(r"Errors\s*=\s*(\d+)", leap_log)):
                    raise RuntimeError(f"tleap reported errors; see {folder / 'preparation_logs' / 'tleap.log'}")
                prepared_path = preparation / "prot_meld_input.pdb"
                prepared = load_pdb(prepared_path)
                require_same_sequence(raw, prepared)
                if "CYX" in prepared["residue_names"]:
                    raise ValueError(f"{run['id']}: preparation detected disulfide chemistry; explicit bond configuration is required.")
                shutil.copy2(prepared_path, folder / "prot_meld_input.pdb")
                run.update(status="prepared", prepared_sha256=sha256(prepared_path))
                write_json(manifest_path, manifest)
            print("Generating the common contact libraries...", flush=True)
            run_logged([sys.executable, TEMPLATES / "generate_contacts.py", "--manifest", manifest_path],
                       output, output / "preparation_logs" / "contacts.log")
            summary = json.loads((output / "reports" / "contact_summary.json").read_text())
            for index, run in enumerate(manifest["runs"]):
                folder = output / run["directory"]
                (folder / "contacts").mkdir(exist_ok=True)
                for name in config["contact_files"].values():
                    shutil.copy2(output / name, folder / name)
                run_logged([sys.executable, TEMPLATES / "pdb_to_fasta.py", "--pdb", folder / "prot_meld_input.pdb",
                            "--raw-pdb", folder / "raw_af2.pdb", "--output-dir", folder],
                           output, folder / "preparation_logs" / "fasta.log")
                for name in ("system_setup.py", "run_meld.sh", "extract_trajectories.sh"):
                    shutil.copy2(TEMPLATES / name, folder / name)
                (folder / "run_meld.sh").chmod(0o755)
                (folder / "extract_trajectories.sh").chmod(0o755)
                run_config = json.loads(json.dumps(config))
                run_config.update(run_id=run["id"], source_structure=run["source_relative"],
                                  n_residues=summary["residue_count"], langevin_seed=config["langevin_seed"] + index)
                write_json(folder / "input.json", run_config)
                job = (TEMPLATES / "job.slurm").read_text().replace("__JOB_NAME__", "meld_" + run["id"][:64])
                if not (folder / "job.slurm").exists():
                    (folder / "job.slurm").write_text(job)
                staged = ["raw_af2.pdb", "prot_meld_input.pdb", "sequence.dat", "sequence.fasta",
                          "residue_map.csv", "input.json", "system_setup.py", "run_meld.sh",
                          *config["contact_files"].values()]
                run["staged_sha256"] = {name: sha256(folder / name) for name in staged}
            manifest.update(status="complete", completed_utc=utc_now(),
                            contact_sha256=summary["contact_sha256"], counts=summary)
            manifest.pop("last_error", None)
            write_json(manifest_path, manifest)
            check_complete(output, manifest)
            print(f"Ready: {len(paths)} model folders; core={summary['core_contacts']}, "
                  f"alternatives={summary['alternative_contacts']}; active={summary['active_counts']}")
            print(f"Output: {output}\nAudit: {output / 'reports' / 'input_restraint_audit.csv'}")
            print("Edit the Slurm samples, then preview with submit_jobs.py. No MD or jobs were launched.")
        except Exception as error:
            manifest["last_error"] = str(error)
            write_json(manifest_path, manifest)
            raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
