#!/usr/bin/env python3
"""Write the exact prepared protein sequence and its ordinal residue map."""

import argparse
from pathlib import Path

from structure_utils import load_pdb, require_same_sequence, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--raw-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = load_pdb(args.pdb)
    raw = load_pdb(args.raw_pdb)
    require_same_sequence(raw, model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequence.dat").write_text(model["sequence"] + "\n")
    (args.output_dir / "sequence.fasta").write_text(">" + args.output_dir.name + "\n" + model["sequence"] + "\n")
    rows = [{"residue_index_1based": i + 1, "residue_name": name,
             "one_letter": model["sequence"][i], "pdb_chain": model["residue_keys"][i][0].strip(),
             "pdb_residue_id": model["residue_keys"][i][1].strip(),
             "raw_residue_id": raw["residue_keys"][i][1].strip(), "raw_plddt": float(raw["plddt"][i])}
            for i, name in enumerate(model["residue_names"])]
    write_csv(args.output_dir / "residue_map.csv", list(rows[0]), rows)
    print(f"{args.output_dir.name}: wrote {len(model['sequence'])} residues")


if __name__ == "__main__":
    main()
