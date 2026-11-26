"""Stream full-atom backbone-aligned merged DCDs and structural exports."""
import csv
from pathlib import Path
import shutil

import mdtraj as md
import numpy as np

from .export_frames import align
from .metrics import dump_json


def aligned_chunks(run, folder, chunk_size):
    """Write every selected slot's frames, even when later analysis trims/strides.

    Fit N/CA/C/O onto this start's prepared input PDB; apply the same proper
    rotation and translation to ALL atoms. No synthetic topology frame is added.
    """
    folder = Path(folder)
    shutil.copy2(run["pdb"], folder / "prot_meld_input.pdb")
    temporary = folder / ".total.writing.dcd"
    map_temp = folder / ".frame_provenance.writing.csv"
    count = 0
    try:
        with md.formats.DCDTrajectoryFile(str(temporary), "w") as writer, map_temp.open("w", newline="") as handle:
            provenance = csv.writer(handle)
            provenance.writerow(["total_frame", "replica", "replica_frame"])
            for item in run["files"]:
                offset = 0
                for chunk in md.iterload(item["source"]["path"], top=run["pdb"], chunk=chunk_size):
                    if not np.isfinite(chunk.xyz).all():
                        raise ValueError(f"Non-finite trajectory coordinates: {item['source']['path']}")
                    frames = np.arange(offset, offset+chunk.n_frames)
                    total_frames = np.arange(count, count+chunk.n_frames)
                    coordinates = align(chunk.xyz, run["indices"], run["reference"].xyz[0, run["indices"]])
                    writer.write((coordinates*10).astype(np.float32))  # low-level DCD API takes Å
                    provenance.writerows(zip(total_frames, np.repeat(item["replica"], len(frames)), frames))
                    offset += chunk.n_frames
                    count += chunk.n_frames
                    backbone = md.Trajectory(coordinates[:, run["indices"]].astype(np.float32), run["backbone_reference"].top)
                    yield item, frames, total_frames, backbone
                if offset != item["frames"]:
                    raise ValueError(f"DCD frame count changed/truncated: {item['source']['path']}")
        temporary.replace(folder / "total.dcd")
        map_temp.replace(folder / "frame_provenance.csv")
        dump_json(folder / "alignment.json", {"status": "complete", "n_frames": count,
                  "n_atoms": run["reference"].n_atoms, "backbone_atom_names": ["N", "CA", "C", "O"],
                  "reference": "prot_meld_input.pdb", "synthetic_initial_frame": False,
                  "order": "ascending selected slot index, then original frame index",
                  "contains_frames_before_analysis_exclusions": True,
                  "selected_slots": [f["replica"] for f in run["files"]], "total_frame_index_base": 0})
    finally:
        temporary.unlink(missing_ok=True)
        map_temp.unlink(missing_ok=True)


def export_reference_structures(run, runs, folder, table, cutoff, write_dcd=True):
    """Closest PDB/JSON and cutoff-selected DCD/CSV for every reference."""
    folder = Path(folder)
    indices = run["indices"]
    for other in runs:
        identifier = other["reference_id"]
        column = identifier + "_rmsd_A"
        selected = table[table[column] <= cutoff].copy()
        selected.to_csv(folder / f"{identifier}_selected_frames.csv", index=False)
        closest = table.loc[table[column].idxmin()]
        frame = md.load_frame(str(folder / "total.dcd"), int(closest.total_frame), top=run["reference"].top)
        frame.xyz = align(frame.xyz, indices, other["reference"].xyz[0, other["indices"]]).astype(np.float32)
        filename = f"closest_to_{identifier}.pdb"
        frame.save_pdb(str(folder / filename))
        dump_json(folder / f"closest_to_{identifier}.json", {"start_model": run["id"],
                  "reference_model": other["id"], "reference_id": identifier,
                  "total_frame": int(closest.total_frame), "replica": int(closest.replica),
                  "replica_frame": int(closest.replica_frame), "rmsd_A": float(closest[column]),
                  "within_main_cutoff": bool(closest[column] <= cutoff), "pdb": filename,
                  "topology": "prot_meld_input.pdb", "aligned_to": other["id"]})
        if write_dcd and len(selected):
            chosen = set(selected.total_frame.astype(int))
            destination = folder / f"{identifier}_cutoff.dcd"
            offset = 0
            with md.formats.DCDTrajectoryFile(str(destination), "w") as writer:
                for chunk in md.iterload(str(folder / "total.dcd"), top=run["pdb"], chunk=500):
                    picks = [i for i in range(chunk.n_frames) if offset+i in chosen]
                    offset += chunk.n_frames
                    if picks:
                        xyz = align(chunk.xyz[picks], indices, other["reference"].xyz[0, other["indices"]])
                        writer.write((xyz*10).astype(np.float32))
