# Portable MELD trajectory analysis

This component analyzes **one protein, any number of prepared starting models**
from this repository's preparation manifest. It has no protein IDs, fixed model
count, fixed replica count, scheduler settings or workstation paths. It neither
imports MELD nor starts simulations. CPU analysis does not require CUDA or VMD.

Read [ANALYSIS.md](ANALYSIS.md) for methods, output definitions and scientific
limits. Agents should also read [AGENTS.md](AGENTS.md).

## Install and run

Run these commands from the repository root. Analysis can use a separate conda
environment, leaving the simulation environment untouched:

```bash
conda env create -f analysis/environment.yml
conda activate meld_analysis
python analyze_trajectories.py --help
```

On an HPC, load its conda/mamba module first if required. The included environment
specification supplies NumPy, MDTraj, pandas, matplotlib and threadpoolctl;
scikit-learn supplies HDBSCAN and optional umap-learn supplies visualization.
Version ranges are dependency constraints, not a fully pinned lock file. Actual
versions and analysis source hashes are recorded in every result manifest.

Finish simulations, then extract/download numbered replicas and the prepared
input files. Retain `pipeline_manifest.json` at the simulation root. For example:

```bash
# Extraction is a separate stage; run in meld_new where MELD is installed.
conda activate meld_new
python extract_all_trajectories.py --simulations simulations --dry-run
python extract_all_trajectories.py --simulations simulations

conda activate meld_analysis
# No analysis output is created by this preflight.
python analyze_trajectories.py --simulations simulations --dry-run

# Structural coverage analysis, without clustering or simulation execution.
python analyze_trajectories.py --simulations simulations --output analysis_results

# Recompute saved assignments and summary accounting; writes validation.json.
python -m analysis.validate_results --analysis analysis_results

# Optional independent second-backend RMSDs for EVERY retained frame.
python -m analysis.validate_results --analysis analysis_results --full-rmsd
```

Custom paths may be absolute and may contain spaces when quoted. Defaults are
relative to the installed repository, not the shell's current working directory.
All files under the input simulation folders remain unchanged. No `total.dcd`
is needed, generated or used. Original numeric DCD frame 0 is a sampled frame.

## Input contract

```text
simulations/
  pipeline_manifest.json          # status=complete, nonempty runs[]
  <any prepared model directory>/
    prot_meld_input.pdb
    input.json                   # includes actual n_replicas and protocol
    contacts/...                 # files referenced by input.json
    setup_report.json            # checked if present, recommended provenance
    0.dcd ... <n_replicas-1>.dcd
```

Discovery uses `runs[].id` and `runs[].directory`, not globbed folder names.
References are all manifest-listed `prot_meld_input.pdb` files. Each reference
is assigned a safe `reference_0000`, `reference_0001`, ... identifier in manifest
order; this is an identifier, not a biological annotation. Original names are
preserved in tables and the reference mapping.

Default behavior requires every configured replica for every start. To analyze
only a requested subset, explicitly use `--replicas 0` or `--replicas 0 1 2`.
Then `pooled` means just those requested indices; it does not imply a full ladder.
The selection is recorded. Missing files, invalid replica indices, differing
sequences/chemistry, differing protocols or contacts, changed staged PDB/config
hashes and topology atom-count mismatches fail preflight. Coordinates must be
protein-only, single-chain, complete N/Cα/C/O backbones with resolved altlocs.
The system should already be contiguous; explicit-solvent periodic imaging is
outside this implicit-solvent pipeline's input contract.

Allowed per-start config differences are `run_id`, `source_structure`, and
`langevin_seed`; other configuration differences stop the shared analysis.
Raw PDB residue numbering and atom order need not match across starts:
the code establishes backbone correspondence by ordered residues and atom name.
Different all-atom topologies are loaded independently, but differing prepared
residue chemistry is rejected for the shared-condition analysis.

Preflight reads references and DCD headers/first coordinates. It does not prove
the remaining DCD records are intact. The actual analysis streams all records,
checks counts/finite coordinates and stops on corrupt/truncated input. A complete
preparation manifest does not prove that simulations finished; check run logs
and extraction completeness separately.

## Cutoffs, windows and computational cost

```bash
# Main default cutoff 4 Å; sensitivity cutoffs 2/3/4/5 Å.
python analyze_trajectories.py --main-cutoff 4 --cutoffs 2 3 4 5 \
  --output analysis_results_cutoffs

# Illustrative equilibration exclusion: replace N with a reviewed frame count.
# Do not assume saved frame number equals MELD cycle number.
python analyze_trajectories.py --discard-frames N --stride 2 \
  --output analysis_results_post_equilibration

# Analyze only replica 0, without implicitly assuming it is equilibrated.
python analyze_trajectories.py --replicas 0 --output analysis_results_replica0

# Optional hashes cost an additional full trajectory read at the start/end.
python analyze_trajectories.py --hash-dcd --output analysis_results_hashed
```

`N` above is a placeholder for a nonnegative integer. Default discard is zero:
the default run is a coverage survey, not an equilibrated population estimate.
`--discard-fractions 0 0.1 0.2` is the default sensitivity analysis. Each fraction
f excludes `floor(original_replica_length*f)` frames independently. The retained
start is `max(--discard-frames, fractional_prefix)`. These are alternative lower
bounds, not two prefixes added together. Stride uses the lattice beginning at
the absolute `--discard-frames` index in each DCD.

RMSDs are streamed in chunks (`--chunk-size 500`) and starts are processed
sequentially. `--threads 1` is the default numerical thread limit. Memory for
RMSD summaries still grows with retained frames × number of references per start.
There is no automatic GPU use. Increase stride/chunk settings deliberately and
use a new output directory for a different resolution.

## Optional structural clustering

```bash
python analyze_trajectories.py --cluster --umap \
  --cluster-max-frames 10000 --output analysis_results_clustered

# Use only replica 0 and a reviewed fractional exclusion for clustering.
python analyze_trajectories.py --cluster --cluster-scope replica_0 \
  --cluster-discard-fraction 0.2 --output analysis_results_cluster_replica0
```

Clustering is supplementary to direct RMSD assignment. Internal soft-contact
and periodic torsion features → PCA retaining 90% variance → HDBSCAN leaf groups;
optional UMAP is for display only. Cluster-size sensitivity and representative
PDBs are included. Parameters are configurable with `--cluster-fraction`,
`--cluster-min-size`, and `--cluster-min-samples`.

To bound cost, at most 2,000 frames per start are sampled, also limited by the
total budget divided across starts. Short starts may provide fewer frames;
the exact sample counts and frame provenance are saved. Selection is deterministic
and uniform over eligible concatenated replica indices within each start.
Clustering does **not** assign unsampled frames. Its `sample_pct` values must not
be substituted for full-trajectory state populations. Dense density clustering
can still consume substantial RAM; begin with a smaller sample budget if needed.
An insufficient or structurally constant sample stops with an explanation;
rerun without clustering or adjust the documented sampling/model settings.

## Outputs, completion and frame export

The output must be a **new directory**. There is no silent overwrite, resume or
cache reuse in the portable component. A failed run remains marked `failed`
(or `running` if interrupted). Inspect it, then choose a new output directory;
do not present a partial report as a completed analysis.

Main outputs:

- `analysis_manifest.json`: status, settings, inputs, hashes/signatures, model
  mapping, protocol, versions, source code hashes, and explicit interpretation flags.
- `reference_rmsd.csv`: all ordered reference-to-reference RMSDs in Å.
- `reference_*/frame_rmsd.csv`: every retained frame, source replica/frame,
  every reference RMSD, main-cutoff assignment and overlap count.
- `reference_populations.csv`: inclusive and exclusive counts for each reference,
  starting model, replica scope, cutoff and discard fraction.
- `assignment_populations.csv`: exclusive reference, ambiguous and unassigned
  categories, which partition the denominator exactly.
- `primary_pooled_summary.csv`, `secondary_replica0_summary.csv`: main cutoff,
  zero additional fractional exclusion; any absolute exclusion still applies.
- `numerical_audit.json`: independent MDTraj/QCP checks of up to three retained
  frames per chunk/reference. This is a sampled audit, not an all-frame second calculation.
- `validation.json`: produced by the explicit validator, with accounting status
  and optional full independent RMSD comparison counts. Check its latest status;
  an existing HTML does not imply a later validation command passed.
- `report.html`: standalone report with embedded figures, summary CSVs and
  provenance/audit downloads. It has no CDN or external figure dependency.
- Optional clustering: `cluster_sample_assignments.csv`,
  `cluster_sample_populations.csv`, `clustering_sensitivity.csv`,
  `clustering_metadata.json`, `cluster_representatives.csv` and full-atom
  `cluster_representatives/*.pdb`.

HTML includes methods, per-start comparisons, exclusive accounting and cross-run
interpretation. It does not embed raw DCDs, every per-frame table, or every
representative PDB. Share the output directory as well when those are needed.

To export full-atom frames within a reference RMSD cutoff, use actual model IDs
from the result manifest (replace the quoted example names):

```bash
python -m analysis.export_frames --analysis analysis_results \
  --start-model 'YOUR_START_MODEL_ID' --reference-model 'YOUR_REFERENCE_MODEL_ID' \
  --cutoff 4 --scope pooled --output exports/matches.dcd
```

Export uses only frames retained by the original analysis's absolute exclusion
and stride. To apply an additional fractional prefix, first analyze that desired
window with an appropriate absolute exclusion or filter the saved frame table
in a separately reviewed workflow. The exporter does not apply the sensitivity
table's fractional windows implicitly.

Each output frame is aligned on N/Cα/C/O to the chosen reference, keeping the
source's full atom order. Load it with the **starting model's topology**, not
necessarily the reference topology. A companion `.frames.csv` records the exact
export order and source indices. No match produces no DCD. Existing exports
are never overwritten. Use `--simulations /new/root` after relocating source
folders; relative manifest paths are preserved. Export verifies reference hashes
and DCD signatures; preserve DCD modification times during copying or originally
analyze with `--hash-dcd` to permit a hash-based check after relocation.

## Component map

| File | Responsibility |
|---|---|
| `../analyze_trajectories.py` | Stable CLI entry point |
| `run_analysis.py` | Manifest validation, streaming and population accounting |
| `metrics.py` | Explicit atom correspondence, proper Kabsch fits, assignments and hashes |
| `cluster.py` | Optional bounded internal-feature clustering and representatives |
| `report.py` | Data-driven plots and standalone HTML |
| `export_frames.py` | Optional full-atom aligned cutoff DCD export |
| `validate_results.py` | Post-run frame/assignment/summary audit and optional all-frame QCP check |
| `environment.yml` | Portable analysis dependencies |
| `../tests/test_analysis.py` | Synthetic integration/scientific checks |

## Tests

```bash
python -m unittest discover -s tests -p test_analysis.py -v
```

Fixtures create small PDBs/DCDs in temporary directories. Tests cover proper
fitting/reflection rejection, overlap, arbitrary start count, one-model input,
replica/frame accounting, prefix/stride behavior, no-write preflight, failed
preflight, existing-output protection, sampled clustering, reporting and aligned
frame export. They do not run MD, call a scheduler or require the example protein.
