"""Matched backbone coordinates and proper-rotation RMSD, in Angstrom."""
import hashlib
import json
from pathlib import Path

import mdtraj as md
import numpy as np

NAMES = ("N", "CA", "C", "O")
CANONICAL = {"CYX": "CYS", "CYM": "CYS", "HID": "HIS", "HIE": "HIS",
             "HIP": "HIS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}
AMINO_ACIDS = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def signature(path, full=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if full:
        result["sha256"] = digest(path)
    return result


def dump_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def backbone(path):
    """Require complete single-chain protein; match residues by ordinal and name.

    Atom order and PDB numbering may differ between prepared reference models.
    Explicit N/CA/C/O ordering excludes OXT and all hydrogens.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if sum(line.startswith("MODEL ") for line in lines) > 1:
        raise ValueError(f"Multiple PDB models: {path}")
    atoms = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    if any(line[16] != " " for line in atoms):
        raise ValueError(f"Resolve alternate locations before analysis: {path}")
    ref = md.load_pdb(str(path), standard_names=False)
    residues = list(ref.top.residues)
    if ref.n_frames != 1 or len(list(ref.top.chains)) != 1 or not residues:
        raise ValueError(f"Expected one protein model/chain: {path}")
    indices, keys, chemistry = [], [], []
    for ordinal, residue in enumerate(residues):
        if CANONICAL.get(residue.name, residue.name) not in AMINO_ACIDS:
            raise ValueError(f"Non-protein residue {residue} in {path}")
        chemistry.append(residue.name)
        for name in NAMES:
            matching = [a.index for a in residue.atoms if a.name == name]
            if len(matching) != 1:
                raise ValueError(f"Expected one {name} in residue {ordinal}: {path}")
            indices.append(matching[0])
            keys.append((ordinal, CANONICAL.get(residue.name, residue.name), name))
    if not np.isfinite(ref.xyz).all():
        raise ValueError(f"Non-finite reference coordinates: {path}")
    return ref, np.asarray(indices), keys, chemistry


def rmsd_A(xyz_nm, reference_nm):
    """Independently fit each frame: unweighted, float64, no mirror reflection."""
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    target = np.asarray(reference_nm, dtype=np.float64)
    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(np.einsum("fni,nj->fij", xyz, target))
    signs = np.ones((len(xyz), 3))
    signs[:, -1] = np.where(np.linalg.det(u @ vt) < 0, -1, 1)
    rotation = (u * signs[:, None, :]) @ vt
    residual = np.einsum("fni,fij->fnj", xyz, rotation) - target
    return np.sqrt(np.mean(np.sum(residual * residual, axis=2), axis=1)) * 10


def matched_backbone(reference, indices):
    """Build explicit canonical atom order; MDTraj atom_slice sorts indices.

    A fresh topology avoids silently returning a different coordinate order
    when an input PDB lists its atoms differently from N/CA/C/O.
    """
    topology = md.Topology()
    chain = topology.add_chain()
    previous_c = None
    for start in range(0, len(indices), 4):
        source = reference.top.atom(int(indices[start])).residue
        residue = topology.add_residue(source.name, chain)
        atoms = [topology.add_atom(reference.top.atom(int(index)).name,
                                   reference.top.atom(int(index)).element, residue)
                 for index in indices[start:start+4]]
        for a, b in [(0, 1), (1, 2), (2, 3)]:
            topology.add_bond(atoms[a], atoms[b])
        if previous_c is not None:
            topology.add_bond(previous_c, atoms[0])
        previous_c = atoms[2]
    return md.Trajectory(reference.xyz[:, indices].copy(), topology)


def assignments(distances, cutoff):
    """Preserve overlapping reference neighborhoods and unmodeled conformations."""
    hits = np.asarray(distances) <= cutoff
    labels = np.full(len(hits), "unassigned", dtype=object)
    counts = hits.sum(axis=1)
    labels[counts > 1] = "ambiguous"
    for index in range(hits.shape[1]):
        labels[(counts == 1) & hits[:, index]] = f"reference_{index:04d}"
    return hits, labels
