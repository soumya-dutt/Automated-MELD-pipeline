# Portable MELD analysis: alignment, sampling and structural clustering

Analyze **one protein with any number of starting structures and configured
replicas**. Inputs come from the preparation manifest. There are no protein IDs,
fixed model/replica counts, scheduler settings or workstation paths in the code.
This is CPU postprocessing: no MELD dynamics, CUDA or VMD is required.

Read [ANALYSIS.md](ANALYSIS.md) for scientific definitions and limits, and
[AGENTS.md](AGENTS.md) for the agent execution contract.

## Install and run

From repository root:

```bash
conda env create -f analysis/environment.yml
conda activate meld_analysis
python analyze_trajectories.py --help
python analyze_trajectories.py --simulations simulations --dry-run
python analyze_trajectories.py --simulations simulations --output analysis_results_new
```

The **full suite runs by default**: backbone alignment and saved `total.dcd`,
multi-reference RMSDs, pooled/slot-0 plots, cutoff/closest-frame exports,
full-frame structural clustering, UMAP and standalone HTML. HDBSCAN needs at
least the configured minimum cluster/sample size and nonconstant structural
features; very short or constant test data may require `--no-cluster`.

For selected reductions in cost:

```bash
# All alignment, RMSD, scatter/sensitivity plots and frame exports; no clustering.
python analyze_trajectories.py --output analysis_results_rmsd --no-cluster

# Explicitly cap the total clustering sample; RMSD analysis still uses all
# retained frames, and aligned total.dcd still includes every source frame.
python analyze_trajectories.py --output analysis_results_sampled \
  --cluster-max-frames 10000 --no-umap

# Skip cutoff-selected DCD exports. Closest PDBs, frame CSVs and total.dcd remain.
python analyze_trajectories.py --output analysis_results_no_cutoff_dcd --no-extract
```

Inputs and simulation folders are never modified. Use a **fresh output directory**;
there is no silent overwrite/resume. No new results are generated merely by
updating this repository's code. Start analysis explicitly when ready.

## Input contract

Finish extraction first (using `extract_all_trajectories.py` in the MELD
environment), then keep this layout locally or on the analysis machine:

```text
simulations/
  pipeline_manifest.json                 # status=complete, nonempty runs[]
  <each manifest-listed model folder>/
    prot_meld_input.pdb
    input.json                          # includes actual n_replicas/protocol
    contacts/...                        # files referenced by input.json
    setup_report.json                   # checked when present
    0.dcd ... <n_replicas-1>.dcd
```

Each model's prepared PDB is a reference. Labels `reference_0000`, ... follow
manifest order; original model IDs are preserved in mapping tables and the HTML.
All configured replicas must exist unless a subset is explicitly requested:

```bash
python analyze_trajectories.py --replicas 0 --output analysis_results_slot0
```

For that example, pooled scope and `total.dcd` contain only the requested slot 0;
the manifest records this restriction. Default selection includes every slot.

Sequences and prepared residue chemistry must agree across starts. Protocols
must agree except for `run_id`, `source_structure` and `langevin_seed`. Shared
contact files must match by hash. PDB/config/staged contact hashes and saved
setup checks prevent silently combining different conditions. Each DCD's atom
count must match its own topology. Require one complete, protein-only chain with
resolved altlocs and exactly one N, CA, C and O per residue. Atom order and PDB
numbering may differ between starts; correspondence is established explicitly.
Periodic imaging and missing-residue alignment are outside this input contract.

The no-write analysis preflight checks references and DCD headers/first frames;
actual analysis streams all coordinates and rejects non-finite or truncated data.
A complete preparation manifest does not prove the simulation finished.

## Alignment and saved merged trajectories

For each starting model, the code:

1. Loads full-atom frames from the selected numbered DCDs in ascending numeric
   slot order, retaining original frame order within each file.
2. Fits each frame's **N/Cα/C/O backbone** to that start's prepared
   `prot_meld_input.pdb`, using proper Kabsch rotation and translation.
3. Applies the same rigid transformation to **all atoms** and writes
   `analysis_results_new/reference_XXXX/total.dcd`.
4. Saves the matching full-atom topology, `frame_provenance.csv` and
   `alignment.json` alongside it.
5. Performs reference comparisons and clustering on the aligned coordinates.
   Every RMSD comparison independently refits to its own reference to ensure
   optimal alignment; internal contacts/torsions are rigid-motion invariant.

**Every frame from every selected replica is saved in `total.dcd`, including
frames later excluded or strided for analysis.** No synthetic initial PDB frame
is added. `total_frame=0` is the first extracted sample; the provenance table
maps every merged frame to its original `replica` and zero-based `replica_frame`.
Load the merged/cutoff DCDs with the **source start's copied topology**.
Different starts' all-atom orders need not agree.

The fit target is the prepared input PDB, not an inferred average structure or
an arbitrary frame from another start. Source coordinates and DCD files remain
unchanged. A merged trajectory is a coverage collection, not a continuous
physical trajectory; jumps between slot blocks are expected.

## Analysis parameters and cost

- Main backbone RMSD cutoff: `--main-cutoff 4` Å.
- Sensitivity: `--cutoffs 2 3 4 5` and `--discard-fractions 0 0.1 0.2`.
- `--discard-frames N`: remove that many original frames from every slot **for
  analysis only**. Replace N with a reviewed nonnegative integer.
- For fractional prefix f, the lower bound is
  `max(N, floor(original_slot_length*f))`; prefixes are not added together.
- `--stride 1`: keep every analysis frame; stride is anchored at N in each slot.
- `--chunk-size 500`, `--threads 1`: coordinate I/O chunks and numerical threads.
- `--hash-dcd`: use full trajectory SHA256 in addition to size/mtime signatures;
  this adds full reads at the beginning/end.
- `--cluster-max-frames 0`: all eligible analysis frames are clustered (default).
  A positive integer caps a deterministic per-start share of the global budget.
- `--cluster-scope pooled|replica_0`, `--cluster-discard-fraction 0`: separately
  record the clustering window. Unsampled/excluded frames are never assigned.
- `--cluster-min-size 40`, `--cluster-min-samples 10`,
  `--cluster-fraction 0.01`: HDBSCAN settings; compare 0.5/1/2% resolution.
- `--no-cluster`, `--no-umap`, `--no-extract`: explicit feature opt-outs.

Full-frame density clustering can use substantial RAM/time. Check sample counts
and available memory before a full run, or choose an explicit sample budget.
The feature-selection step uses up to 1,000 frames per start for balanced contact
variance selection; that cap does **not** cap the default clustering assignments.
All reference pairs are plotted by default, so plot count grows with model count.
Full-atom merged files and cutoff DCDs need additional disk space; overlapping
reference matches can duplicate exported coordinates.

Default exclusion is zero frames. These defaults produce an untrimmed coverage
survey, not a declaration of equilibration or a final population rank order.

## Outputs and correspondence with the original pair analysis

| Output | What it contains |
|---|---|
| `reference_*/total.dcd`, `prot_meld_input.pdb` | Backbone-aligned full-atom frames and source topology |
| `reference_*/frame_provenance.csv`, `alignment.json` | All original slot/frame → merged-frame mappings and alignment definition |
| `reference_*/frame_rmsd.csv` | Retained analysis frames, original and merged indices, all reference RMSDs and assignments |
| `pairwise/<refA>__<refB>/rmsd_scatter_pooled.png` | Pooled backbone RMSD scatter to two references, panels for every start |
| `pairwise/.../rmsd_scatter_replica_0.png` | Same comparison using only slot 0 |
| `pairwise/.../two_reference_rmsd*.png` | Original-style log-count density plots with pooled/slot-0 rows, paginated over starts |
| `reference_*/rmsd_distributions.png` | Backbone RMSD distributions to all references, pooled and slot 0 |
| `reference_*/replica_0_rmsd_trace.png` | All reference RMSDs against original slot-0 frame index |
| `reference_*/reference_matches_by_replica.png` | Inclusive occupancy at every selected temperature/alpha slot |
| `reference_*/cutoff_sensitivity.png`, `discard_sensitivity.png` | RMSD and per-slot prefix sensitivity |
| `reference_*/reference_XXXX_selected_frames.csv`, `_cutoff.dcd` | Main-cutoff selected frames for every reference; zero matches create no DCD |
| `reference_*/closest_to_reference_XXXX.pdb`, `.json` | Closest full-atom frame to each reference, aligned to it, with provenance; closest need not meet cutoff |
| `reference_populations.csv`, `assignment_populations.csv` | Every start/slot/cutoff/window; inclusive and exclusive/ambiguous/unassigned accounting |
| `primary_pooled_summary.csv`, `secondary_replica0_summary.csv` | Main cutoff, no additional fractional exclusion; absolute exclusion still applies |
| `pairwise_reference_populations.csv` | A-only/B-only/both/neither for every plotted reference pair and start/scope |
| `rmsd_statistics.csv`, `closest_reference_frames.csv`, `sampling_overview.csv` | Quantiles, closest-frame indices and cross-start summary |
| `cluster_assignments.csv`, `cluster_populations.csv` | Common HDBSCAN labels and per-start pooled/slot-0 populations, with full-versus-sampled status |
| `cluster_sample_assignments.csv`, `cluster_sample_populations.csv` | Compatibility tables explicitly describing the actual clustered input |
| `clustering_metadata.json`, `clustering_sensitivity.csv`, `clustering_overview.csv` | Feature/PCA settings, resolution sensitivity, noise and source-slot association |
| `cluster_source_diagnostics.csv` | Each group's distribution over starts and source slots |
| `cluster_representatives.csv`, `cluster_representatives/*.pdb` | Actual aligned full-atom sampled frames nearest cluster centroids |
| `structural_clusters_umap.png`, `cluster_populations.png`, `cluster_population_heatmap.png`, `clustering_sensitivity.png` | Structural embedding, top/all groups, noise and sensitivity |
| `RESULTS.md`, `report.html` | Readable results and standalone methods/pair-comparison/start-comparison/conclusion report |
| `analysis_manifest.json`, `numerical_audit.json`, `validation.json` | Input/config/version provenance, sampled numerical audit, explicit validation results |

Clustering labels are fitted jointly across starts of this one protein. Default
full-frame populations cover all retained analysis frames; explicit sample/window
restrictions are reported and must accompany their percentages. Noise is retained.
UMAP is visualization only, not the clustering metric or an RMSD assignment rule.

## Validation and additional exports

```bash
# Accounting, merged-frame provenance/orientation samples, cutoff counts and closest metadata.
python -m analysis.validate_results --analysis analysis_results_new

# Also recalculate every retained source/reference RMSD independently with MDTraj/QCP.
python -m analysis.validate_results --analysis analysis_results_new --full-rmsd

# Export an additional cutoff selection. Replace model IDs with those in the manifest.
python -m analysis.export_frames --analysis analysis_results_new \
  --start-model 'YOUR_START_MODEL_ID' --reference-model 'YOUR_REFERENCE_MODEL_ID' \
  --cutoff 3 --scope replica_0 --output exports/additional_matches.dcd
```

Full independent validation and additional export currently read original source
DCDs; use `--simulations /new/root` after relocation. The validator records actual
comparison counts. A passed sampled audit does not imply an all-frame second
calculation. Check `analysis_manifest.json: status=complete` and validation status.

## Optional report-only regeneration

When explicitly requested, plots/HTML can be regenerated from existing RMSD CSVs
without rereading DCDs, rebuilding alignment or rerunning clustering:

```bash
python -m analysis.report --analysis /path/to/downloaded_analysis
python -m analysis.report --analysis /path/to/downloaded_analysis --discard-fraction 0.2
python -m analysis.report --analysis /path/to/downloaded_analysis \
  --reference-pairs reference_0000:reference_0001
```

This is an explicit write action: it refreshes derived plots/tables/HTML in that
directory and backs up its existing HTML under `report_backups/`. Original
frame-level RMSD/population inputs and the analysis manifest are unchanged.
`report_manifest.json` records report provenance. A report rebuild cannot create
missing trajectories, structural features, clusters or representative PDBs.
The HTML embeds its displayed figures, summary downloads and available
reference/closest/cluster representative PDBs. Full DCDs and per-frame tables
remain companion files.

## Components and tests

`../analyze_trajectories.py` calls `run_analysis.py`; `metrics.py` defines matching
and fitting, `trajectories.py` handles aligned totals and reference exports,
`cluster.py` handles joint structural clustering, `plotting.py` creates diagnostic
figures, and `report.py` compiles HTML. `export_frames.py` and
`validate_results.py` provide explicit follow-up commands. All are contained in
this project and have no dependency on the historical workspace.

```bash
python -m unittest discover -s tests -p test_analysis.py -v
```

Tests use temporary synthetic PDB/DCD files. They cover proper rotations,
reflection rejection, backbone correspondence, full-atom saved alignment,
all-replica merging, prefix/stride provenance, overlap, cutoff exports,
full/sampled clustering, diagnostics and report-only operation without source
paths. They never run dynamics, submit jobs or modify a user's existing results.
