#!/usr/bin/env python3
"""Create shared/alternative C-alpha contact libraries for any model count."""

import argparse
import json
from pathlib import Path

import numpy as np

from structure_utils import (CONTACT_FIELDS, load_pdb, manifest_runs,
                             require_same_sequence, sha256, write_csv, write_json)
from system_setup import active_count, audit_collection


def generate(manifest_path):
    root, manifest = manifest_runs(manifest_path)
    config = manifest["protocol"]
    settings = config["contact_generation"]
    models, raw_models = [], []
    for run in manifest["runs"]:
        folder = root / run["directory"]
        prepared = load_pdb(folder / "prot_meld_input.pdb")
        raw = load_pdb(folder / "raw_af2.pdb")
        require_same_sequence(raw, prepared)
        if models:
            require_same_sequence(models[0], prepared)
            if prepared["residue_names"] != models[0]["residue_names"]:
                raise ValueError("Prepared models differ in residue chemistry/protonation; standardize them first.")
        models.append(prepared)
        raw_models.append(raw)
    nres = len(models[0]["sequence"])
    i, j = np.triu_indices(nres, int(settings["minimum_sequence_separation"]))
    counts = np.zeros(len(i), dtype=int)
    minimum = np.full(len(i), np.inf)
    maximum = np.full(len(i), -np.inf)
    cutoff = settings["cutoff_A"]
    # Stream over models: memory does not scale as models * residues^2.
    for model in models:
        distance = np.linalg.norm(model["ca_A"][i] - model["ca_A"][j], axis=-1)
        counts += distance <= cutoff
        minimum = np.minimum(minimum, distance)
        maximum = np.maximum(maximum, distance)
    min_confidence = np.minimum.reduce([model["plddt"] for model in raw_models])
    core_mask = ((counts == len(models)) & (maximum - minimum <= settings["core_maximum_spread_A"])
                 & (min_confidence[i] >= settings["minimum_core_plddt"])
                 & (min_confidence[j] >= settings["minimum_core_plddt"]))
    union = counts > 0
    alternative_mask = union & ~core_mask
    if not union.any():
        raise ValueError("No contacts meet the cutoff and sequence separation; adjust contact settings explicitly.")
    members = [[] for _ in range(int(union.sum()))]
    ui, uj = i[union], j[union]
    for run, model in zip(manifest["runs"], models):
        distance = np.linalg.norm(model["ca_A"][ui] - model["ca_A"][uj], axis=-1)
        for index in np.flatnonzero(distance <= cutoff):
            members[index].append(run["id"])
    member_lookup = dict(zip(np.flatnonzero(union), members))
    records = {"core": [], "alternatives": []}
    for label, mask in (("core", core_mask), ("alternatives", alternative_mask)):
        for p in np.flatnonzero(mask):
            if label == "core":
                low = max(0.0, (minimum[p] - settings["core_margin_A"]) / 10)
                high = (maximum[p] + settings["core_margin_A"]) / 10
            else:
                low, high = 0.0, settings["alternative_upper_bound_A"] / 10
            records[label].append({
                "residue_i": int(i[p] + 1), "atom_i": "CA",
                "residue_j": int(j[p] + 1), "atom_j": "CA",
                "r1_nm": max(0.0, low - 0.1), "r2_nm": low,
                "r3_nm": high, "r4_nm": high + 0.1,
                "k_kj_mol_nm2": settings["force_constant_kj_mol_nm2"],
                "models_with_contact": int(counts[p]), "contact_models": ";".join(member_lookup[p]),
                "minimum_pair_plddt": float(min(min_confidence[i[p]], min_confidence[j[p]])),
                "distance_spread_A": float(maximum[p] - minimum[p]),
            })
    contacts_dir = root / "contacts"
    reports = root / "reports"
    contacts_dir.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    filenames = config["contact_files"]
    for label in records:
        write_csv(root / filenames[label], CONTACT_FIELDS, records[label])
    audits = []
    selected_fraction = config["collections"]["alternatives"]["confidence"]
    for run, model in zip(manifest["runs"], models):
        core = audit_collection(model["ca_A"] / 10, records["core"], config["collections"]["core"]["confidence"])
        for fraction in sorted({0.60, 0.65, 0.70, selected_fraction}):
            alternative = audit_collection(model["ca_A"] / 10, records["alternatives"], fraction)
            audits.append({"model": run["id"], "core_confidence": core["confidence"],
                           "core_active": core["active_contacts"],
                           "core_bias_kj_mol": core["full_strength_bias_kj_mol"],
                           "alternative_confidence": fraction,
                           "alternative_active": alternative["active_contacts"],
                           "alternative_zero_energy": alternative["zero_energy_contacts"],
                           "alternative_violated_active": alternative["minimum_violated_active_contacts"],
                           "alternative_bias_kj_mol": alternative["full_strength_bias_kj_mol"]})
    write_csv(reports / "input_restraint_audit.csv", list(audits[0]), audits)
    summary = {
        "model_count": len(models), "residue_count": nres,
        "union_contacts": int(union.sum()), "shared_before_filters": int((counts == len(models)).sum()),
        "core_contacts": len(records["core"]), "alternative_contacts": len(records["alternatives"]),
        "active_counts": {label: active_count(config["collections"][label]["confidence"], len(records[label]))
                          for label in records},
        "contact_sha256": {name: sha256(root / name) for name in filenames.values()},
        "settings": settings,
        "notes": [
            "Counts change with the input ensemble; fixed confidence does not imply a fixed number of restraints.",
            "Audit values are analytic full-strength restraint energies, not populations or force-field energies.",
            "An empty collection is omitted from MELD; its CSV and audit remain present.",
        ],
    }
    if not records["core"]:
        summary["notes"].append("No shared core contacts passed the filters; only the alternatives collection will guide MELD.")
    if len(models) == 1:
        summary["notes"].append("One starting model: no multi-model population comparison is available.")
    write_json(reports / "contact_summary.json", summary)
    print(json.dumps({key: summary[key] for key in
                      ("model_count", "core_contacts", "alternative_contacts", "active_counts")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    generate(parser.parse_args().manifest)
