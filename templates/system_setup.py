#!/usr/bin/env python3
"""Build and validate a normal-MELD data store; never launch dynamics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np


def active_count(confidence, total):
    value = Decimal(str(confidence))
    if not Decimal("0") <= value <= Decimal("1"):
        raise ValueError("Confidence must be between zero and one.")
    return int((value * total).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_contacts(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"residue_i", "residue_j", "r1_nm", "r2_nm", "r3_nm", "r4_nm", "k_kj_mol_nm2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing contact columns in {path}")
        contacts = list(reader)
    seen = set()
    for contact in contacts:
        for key in ("residue_i", "residue_j"):
            contact[key] = int(contact[key])
        for key in ("r1_nm", "r2_nm", "r3_nm", "r4_nm", "k_kj_mol_nm2"):
            contact[key] = float(contact[key])
        pair = contact["residue_i"], contact["residue_j"]
        if pair[0] < 1 or pair[0] >= pair[1] or pair in seen:
            raise ValueError(f"Invalid or duplicate contact in {path}: {pair}")
        seen.add(pair)
        bounds = [contact[f"r{index}_nm"] for index in range(1, 5)]
        if not all(np.isfinite(bounds)) or bounds[0] < 0 or bounds != sorted(bounds):
            raise ValueError(f"Invalid distance bounds in {path}: {pair}")
        if not np.isfinite(contact["k_kj_mol_nm2"]) or contact["k_kj_mol_nm2"] < 0:
            raise ValueError(f"Invalid force constant in {path}: {pair}")
    return contacts


def distance_energy(distance, contact):
    """MELD flat-bottom, quadratic shoulders, linear tails; nm and kJ/mol."""
    r1, r2, r3, r4 = (contact[f"r{i}_nm"] for i in range(1, 5))
    k = contact["k_kj_mol_nm2"]
    if distance < r1:
        return 0.5 * k * (r1 - r2) ** 2 + k * (r1 - r2) * (distance - r1)
    if distance < r2:
        return 0.5 * k * (distance - r2) ** 2
    if distance <= r3:
        return 0.0
    if distance < r4:
        return 0.5 * k * (distance - r3) ** 2
    return 0.5 * k * (r4 - r3) ** 2 + k * (r4 - r3) * (distance - r4)


def audit_collection(ca_nm, contacts, confidence):
    energies = np.array([
        distance_energy(float(np.linalg.norm(
            ca_nm[c["residue_i"] - 1] - ca_nm[c["residue_j"] - 1]
        )), c) for c in contacts
    ])
    count = active_count(confidence, len(contacts))
    zero = int(np.count_nonzero(energies <= 1e-12))
    return {
        "confidence": confidence,
        "total_contacts": len(contacts),
        "active_contacts": count,
        "zero_energy_contacts": zero,
        "minimum_violated_active_contacts": max(0, count - zero),
        "full_strength_bias_kj_mol": float(np.sort(energies)[:count].sum()),
    }


def archive_existing(run_dir):
    paths = [run_dir / name for name in ("Data", "Logs", "setup_report.json", "setup.log")]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = run_dir / "setup_backups" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(destination / path.name))
    print(f"Archived previous setup in {destination}")


def validate_saved_setup(run_dir):
    """Read serialized objects and netCDF stage zero without running a runner."""
    from meld import vault
    import netCDF4

    report_path = run_dir / "setup_report.json"
    if not report_path.is_file():
        raise RuntimeError("Missing setup_report.json; setup may not have completed.")
    report = json.loads(report_path.read_text())
    effective = report["effective_config"]
    for name, expected in report["input_sha256"].items():
        if sha256(run_dir / name) != expected:
            raise RuntimeError(f"Input changed since Data was built: {name}")
    if json.loads((run_dir / "input.json").read_text()) != effective:
        raise RuntimeError("input.json changed since Data was built; archive and regenerate.")
    store = vault.DataStore.load_data_store()
    system = store.load_system()
    options = store.load_run_options()
    runner = store.load_remd_runner()
    communicator = store.load_communicator()
    expected_atoms = report["built_atom_count"]
    expected_replicas = effective["n_replicas"]
    assert store.n_atoms == system.n_atoms == expected_atoms
    assert store.n_replicas == runner.n_replicas == communicator.n_replicas == expected_replicas
    assert options.timesteps == effective["timesteps_per_cycle"]
    assert runner.max_steps == effective["max_cycles"]
    assert runner.step >= 1
    assert (run_dir / "Logs").is_dir()
    collections = system.restraints.selectively_active_collections
    labels = [name for name in ("core", "alternatives") if report["restraints"][name]["total_contacts"]]
    assert len(collections) == len(labels)
    for collection, label in zip(collections, labels):
        expected = report["restraints"][label]
        assert len(collection.groups) == expected["total_contacts"]
        assert collection.num_active == expected["active_contacts"]
        assert all(group.num_active == 1 and len(group.restraints) == 1 for group in collection.groups)
    with netCDF4.Dataset(run_dir / "Data" / "Blocks" / "block_000000.nc", "r") as data:
        if len(data.dimensions["timesteps"]) < 1:
            raise RuntimeError("Missing initialized stage-zero coordinates.")
        positions = np.asarray(data.variables["positions"][:, :, :, 0])
        assert positions.shape == (expected_replicas, expected_atoms, 3)
        assert np.isfinite(positions).all()
        assert np.allclose(positions[0], system.template_coordinates, atol=1e-5)
        assert np.allclose(positions, positions[0], atol=1e-5)
    return {"status": "validated", "run": run_dir.name, "atoms": expected_atoms,
            "replicas": expected_replicas, "runner_next_cycle": runner.step,
            "Data_exists": True, "Logs_exists": True}


def build_setup(run_dir, config):
    import meld
    from meld import comm, vault
    from meld.remd import adaptor, ladder, leader
    from openmm import unit as u

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "setup.log")],
        force=True,
    )
    log = logging.getLogger("meld_setup")
    contacts = {name: read_contacts(run_dir / path) for name, path in config["contact_files"].items()}
    with (run_dir / "residue_map.csv").open(newline="") as handle:
        residue_map = list(csv.DictReader(handle))
    core_pairs = {(c["residue_i"], c["residue_j"]) for c in contacts["core"]}
    alternative_pairs = {(c["residue_i"], c["residue_j"]) for c in contacts["alternatives"]}
    if core_pairs & alternative_pairs:
        raise ValueError("Core and alternative contact collections overlap.")
    if not core_pairs and not alternative_pairs:
        raise ValueError("No eligible contacts in either collection.")
    if config["n_replicas"] < 2 or config["max_cycles"] <= config["equilibration_cycles"]:
        raise ValueError("Need at least two replicas and production after equilibration.")
    if not 0 < config["ramp_end_cycle"] <= config["equilibration_cycles"]:
        raise ValueError("The restraint ramp must end within equilibration.")

    subsystem = meld.AmberSubSystemFromPdbFile(config["pdb_file"])
    amber = meld.AmberOptions(
        forcefield=config["forcefield"],
        implicit_solvent_model=config["implicit_solvent_model"],
        use_big_timestep=False,
        use_bigger_timestep=False,
        cutoff=config["nonbonded_cutoff_nm"] * u.nanometer,
        remove_com=False,
        enable_amap=False,
    )
    system = meld.AmberSystemBuilder(amber).build_system([subsystem]).finalize()
    system.integrator.setRandomNumberSeed(config["langevin_seed"])
    residues = list(system.topology.residues())
    aliases = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS"}
    if len(residues) != len(residue_map):
        raise ValueError("Amber changed the number of residues.")
    for actual, expected in zip(residues, residue_map):
        a = aliases.get(actual.name, actual.name)
        b = aliases.get(expected["residue_name"], expected["residue_name"])
        if a != b:
            raise ValueError(f"Residue identity mismatch: {actual.index + 1}: {a} != {b}")
    ca_indices = [system.index.atom(index, "CA", expected_resname=residue.name)
                  for index, residue in enumerate(residues)]
    ca_nm = system.template_coordinates[np.asarray(ca_indices, dtype=int)]
    system.temperature_scaler = meld.GeometricTemperatureScaler(
        0.0, 1.0, config["temperature_min_K"] * u.kelvin,
        config["temperature_max_K"] * u.kelvin,
    )
    ramp = system.restraints.create_scaler(
        "nonlinear_ramp", start_time=1, end_time=config["ramp_end_cycle"],
        start_weight=0.001, end_weight=1.0, factor=4.0,
    )
    audits = {}
    for label in ("core", "alternatives"):
        settings = config["collections"][label]
        scaler = system.restraints.create_scaler(
            "nonlinear", alpha_min=settings["alpha_min"], alpha_max=settings["alpha_max"],
            factor=settings["scaler_factor"], strength_at_alpha_min=1.0,
            strength_at_alpha_max=0.0,
        )
        restraints = []
        for contact in contacts[label]:
            i, j = contact["residue_i"] - 1, contact["residue_j"] - 1
            if j >= len(residues):
                raise ValueError("Contact residue index exceeds the built topology.")
            rest = system.restraints.create_restraint(
                "distance", scaler, ramp,
                atom1=ca_indices[i], atom2=ca_indices[j],
                **{f"r{n}": contact[f"r{n}_nm"] * u.nanometer for n in range(1, 5)},
                k=contact["k_kj_mol_nm2"] * u.kilojoule_per_mole / u.nanometer**2,
            )
            restraints.append(rest)
        confidence = settings["confidence"]
        if restraints:
            system.restraints.add_selectively_active_collection(
                restraints, active_count(confidence, len(restraints)),
            )
        audits[label] = audit_collection(ca_nm, contacts[label], confidence)
        log.info("%s: %s", label, audits[label])
    options = meld.RunOptions(timesteps=config["timesteps_per_cycle"], minimize_steps=2000)
    replicas = config["n_replicas"]
    states = []
    for index in range(replicas):
        state = system.get_state_template()
        state.positions = state.positions.copy()
        state.velocities = state.velocities.copy()
        state.alpha = index / (replicas - 1)
        states.append(state)
    store = vault.DataStore(states[0], replicas, system.get_pdb_writer(), block_size=50)
    try:
        # MELD itself creates BOTH directories. Do not create Logs beforehand.
        store.initialize(mode="w")
        store.save_system(system)
        store.save_run_options(options)
        policy = adaptor.AdaptationPolicy(2.0, 50, 50, stop_after=config["equilibration_cycles"])
        adaptation = adaptor.EqualAcceptanceAdaptor(replicas, policy, min_acc_prob=0.02)
        exchanges = ladder.NearestNeighborLadder(n_trials=replicas * replicas)
        runner = leader.LeaderReplicaExchangeRunner(
            replicas, config["max_cycles"], exchanges, adaptation,
        )
        store.save_remd_runner(runner)
        store.save_communicator(comm.MPICommunicator(system.n_atoms, replicas, timeout=60000))
        store.save_states(states, 0)
        store.save_data_store()
    finally:
        store.close()
    inputs = [config["pdb_file"], "sequence.dat", "residue_map.csv", "input.json",
              "system_setup.py", *config["contact_files"].values()]
    versions = {name: importlib.metadata.version(name)
                for name in ("meld", "openmm", "numpy", "netCDF4", "mpi4py")}
    report = {
        "status": "setup_complete_no_dynamics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "effective_config": config,
        "software_versions": versions,
        "python_version": sys.version,
        "built_atom_count": system.n_atoms,
        "built_residue_count": len(residues),
        "integrator_timestep_fs": system.integrator.getStepSize().value_in_unit(u.femtosecond),
        "restraints": audits,
        "input_sha256": {name: sha256(run_dir / name) for name in inputs},
        "dynamics_steps_executed": 0,
        "saved_coordinate_stage": 0,
        "analysis_first_cycle": config["equilibration_cycles"] + 1,
        "notes": [
            "Input bias audit is analytic and is evaluated at full force constant and ramp=1.",
            "Saved stage-zero energy values are MELD placeholders, not evaluated force-field energies.",
            "The Langevin seed differs by starting structure; initial velocities follow the MELD template.",
        ],
    }
    (run_dir / "setup_report.json").write_text(json.dumps(report, indent=2) + "\n")
    checked = validate_saved_setup(run_dir)
    report["validation"] = checked
    (run_dir / "setup_report.json").write_text(json.dumps(report, indent=2) + "\n")
    log.info("Setup validated. No minimization, dynamics, or replica exchange was executed.")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-existing", action="store_true",
                        help="Preserve an existing Data/Logs setup in setup_backups before rebuilding.")
    parser.add_argument("--check", action="store_true", help="Validate an existing setup without rebuilding.")
    args = parser.parse_args()
    run_dir = Path(__file__).resolve().parent
    os.chdir(run_dir)
    if args.check:
        print(json.dumps(validate_saved_setup(run_dir), indent=2))
        return
    config = json.loads((run_dir / "input.json").read_text())
    if args.archive_existing:
        archive_existing(run_dir)
    elif any((run_dir / name).exists() for name in ("Data", "Logs")):
        raise RuntimeError("Data or Logs already exists. Use --check, or --archive-existing to rebuild.")
    print(json.dumps(build_setup(run_dir, config), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # MELD imports install an MPI exception hook. Setup errors need no MPI abort.
        traceback.print_exc()
        sys.exit(1)
