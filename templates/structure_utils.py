"""Shared file/structure utilities for preparation; no MELD or GPU initialization."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

AA = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
))
AA.update(HIE="H", HID="H", HIP="H", CYX="C", ASH="D", GLH="E", LYN="K")
CONTACT_FIELDS = ["residue_i", "atom_i", "residue_j", "atom_j", "r1_nm", "r2_nm",
                  "r3_nm", "r4_nm", "k_kj_mol_nm2", "models_with_contact",
                  "contact_models", "minimum_pair_plddt", "distance_spread_A"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pdb(path):
    """Read one standard-amino-acid protein chain, preserving ordinal mapping."""
    path = Path(path)
    lines = path.read_text().splitlines()
    if sum(line.startswith("MODEL ") for line in lines) > 1:
        raise ValueError(f"{path.name}: split multiple MODEL records into separate PDB files.")
    if any(line.startswith("HETATM") for line in lines):
        raise ValueError(f"{path.name}: protein-only ATOM records are required; HETATM needs separate preparation.")
    atoms = [line for line in lines if line.startswith("ATOM  ")]
    if any(line[16] not in (" ", "A") for line in atoms):
        raise ValueError(f"{path.name}: resolve alternative atom locations first.")
    ca = [line for line in atoms if line[12:16].strip() == "CA"]
    if not ca:
        raise ValueError(f"{path.name}: no CA atoms were found.")
    keys = [(line[21], line[22:27]) for line in ca]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path.name}: duplicate CA atoms/residue identifiers.")
    if len({key[0] for key in keys}) != 1:
        raise ValueError(f"{path.name}: this pipeline expects a single protein chain.")
    atom_residues = {(line[21], line[22:27]) for line in atoms}
    if atom_residues != set(keys):
        raise ValueError(f"{path.name}: at least one residue is missing its CA atom.")
    names = [line[17:20].strip() for line in ca]
    unknown = set(names) - set(AA)
    if unknown:
        raise ValueError(f"{path.name}: unsupported residues: {sorted(unknown)}")
    xyz = np.array([[float(s[30:38]), float(s[38:46]), float(s[46:54])] for s in ca])
    plddt = np.array([float(s[60:66]) for s in ca])
    if not np.isfinite(xyz).all() or not np.isfinite(plddt).all():
        raise ValueError(f"{path.name}: non-finite coordinates or B-factors.")
    return {"path": path, "sequence": "".join(AA[name] for name in names),
            "residue_names": names, "residue_keys": keys, "ca_A": xyz,
            "plddt": plddt, "atom_count": len(atoms)}


def require_same_sequence(reference, model):
    if reference["sequence"] != model["sequence"]:
        raise ValueError(
            f"Sequence mismatch: {reference['path'].name} ({len(reference['sequence'])} residues) "
            f"versus {model['path'].name} ({len(model['sequence'])} residues). "
            "Use a separate pipeline output for each protein; full sequence correspondence is required."
        )


def manifest_runs(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    root = path.parent
    for run in manifest["runs"]:
        resolved = (root / run["directory"]).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_dir():
            raise ValueError(f"Run directory is outside/missing from the project: {run['directory']}")
    return root, manifest
