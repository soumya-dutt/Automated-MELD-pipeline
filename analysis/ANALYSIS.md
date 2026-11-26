# General analysis methods and interpretation

## Execution contract

This is the reusable analysis component for the automated MELD preparation and
extraction pipeline. It supports **one protein sequence, N starting models**,
where N can be one or more, and uses each model's configured replica count.
No biological identities, special input structures, historical results, fixed
replica numbers, absolute machine paths or scheduler logic are encoded here.

Entry points are `../analyze_trajectories.py`,
`python -m analysis.export_frames`, `python -m analysis.validate_results`, and
the explicit CSV-only `python -m analysis.report` command.
All supporting analysis code is in this
directory. Invocation examples, dependencies and output schemas are in
[README.md](README.md). This component does not need the old `meld_analysis/`
directory, the source simulation codebase, or legacy VMD scripts at runtime.

Stages are: validate manifest and source structures → validate replica files →
backbone-align full-atom coordinates and save merged DCDs → compute matched RMSDs →
assign cutoff neighborhoods → summarize each start and replica → describe
structural diversity (unless disabled) → compile an offline HTML
report. A completed result requires `analysis_manifest.json: status=complete`.
Successful preparation and completed analysis do not by themselves establish
that dynamics converged or that a saved replica is a particular physical ensemble.

## Inputs and structural correspondence

The complete preparation manifest supplies the list of model folders. Each
model's `input.json` supplies the replica count and scientific protocol;
`prot_meld_input.pdb` supplies its reference and trajectory topology. All numeric
DCDs requested by the CLI must exist and match that topology's atom count.
Contact files must agree by SHA256 across starting conditions. Staged PDB/config
hashes are checked when available, and a present setup report must agree with
the current configuration. Protocol differences other than model identity,
source filename and Langevin seed stop analysis.

Residues are matched by ordinal and canonical amino-acid name. Within each
residue the code explicitly finds N, CA, C and O. This allows different PDB
atom ordering and residue numbering without relying on raw atom indices across
files. Each residue must contain exactly one of each selected atom. Explicit
alternative locations, multiple models/chains, missing backbone atoms and
nonprotein residues stop analysis. Prepared residue-name differences, including
protonation/disulfide annotations, also stop shared-condition analysis. Chemistry
requiring a different comparison should be treated as a separate condition.

Topology and reference PDBs are loaded with each trajectory independently;
no all-atom correspondence between different starts is assumed. Sequence
alignment, missing-residue imputation and domain-specific fits are not inferred.
This is a complete-backbone comparison of the preparation pipeline's inputs.

## RMSD metric

First, each source frame is aligned on N/CA/C/O to its own start's prepared
input PDB. A single proper Kabsch transform is applied to all atoms. The aligned
full-atom coordinates are streamed to `reference_*/total.dcd` before analysis
exclusions/stride; the source DCDs are never changed. The copied topology,
`alignment.json`, and `frame_provenance.csv` make the alignment and merged order
explicit. This step uses no VMD and adds no synthetic initial frame.

Each retained frame is independently superposed on every reference using
unweighted float64 Kabsch/SVD fitting, with translation removed and determinant
+1 rotations enforced. The fit and RMSD both use the same N/CA/C/O atoms.
Hydrogens, side chains and terminal OXT are excluded. The RMSD is the square
root of the average squared residual distance across those matched atoms.
MDTraj stores nm internally; reported distances multiply by 10 to give Å.

A global fit measures overall backbone similarity. It does not isolate domain
motion, side-chain rotamers, local active-site geometry or secondary structure
content. A globally dissimilar frame can still contain well-folded domains.
No automatic interpretation as folding/unfolding follows from this metric.

Independently computed MDTraj/QCP RMSDs audit up to three retained frames from
each chunk against every reference. A discrepancy greater than 0.025 Å fails
the run. The saved audit states its sampling scope. All reported RMSDs use
Kabsch, including values near cutoff boundaries. This portable audit is less
exhaustive than the old study's separate all-frame second-backend validation.
The explicit validator recomputes assignments and summary accounting from the
saved frame tables. Its optional `--full-rmsd` mode streams all retained source
frames and checks them with MDTraj/QCP, with a 0.025 Å discrepancy threshold.
It records actual comparison counts and writes `validation.json`; it does not
silently replace the reported Kabsch values or assert all-frame validation
when run in accounting-only mode.
For the new merged outputs it also checks frame counts and original indices,
samples the saved backbone orientation without refitting, and verifies cutoff
exports and closest-frame metadata.

## Assignment to multiple reference neighborhoods

Defaults: main cutoff 4 Å and sensitivity cutoffs 2, 3, 4, 5 Å. These are
configurable analysis choices, not universal definitions of conformational states.
Membership includes equality at the cutoff.

For N references and each frame, compute the N independent RMSDs:

- Inclusive occupancy of reference j counts every frame with RMSD(j) ≤ cutoff.
- An exclusive reference-j assignment requires that j is the only reference
  within cutoff.
- `ambiguous` requires two or more references within cutoff.
- `unassigned` requires none within cutoff.

Exclusive reference assignments + ambiguous + unassigned sum to all retained
frames. Inclusive reference percentages may sum to more than 100%. A nearest
reference is not forced onto ambiguous or unassigned frames. All reference-pair
separations are exported so overlapping definitions can be reviewed explicitly.
Even one model is supported; it provides self-reference coverage but no
between-conformation comparison. With N>2 the procedure remains unchanged.

## Replica provenance and windows

Numbered DCD files are authoritative. `replica_frame=0` is the first actual
sample in that extracted DCD; there is no synthetic topology frame to remove.
No preexisting `total.dcd` is needed. The code creates a new aligned total for
each start inside its analysis output directory. All selected numeric DCDs enter
in ascending slot order, with no trimming or striding of the saved merged file.
Its zero-based `total_frame` maps back to the original slot/frame for every
sample. Burn-in/stride apply to the subsequent analysis tables, not that file.
Independently fitted RMSDs and internal feature values are invariant to this
initial rigid alignment. The merged path is not a continuous kinetic trajectory.

`--discard-frames D` omits the first D records from every selected replica.
Stride is applied on original indices relative to D. For each sensitivity
fraction f, a further lower bound is `floor(original_replica_length*f)`;
the actual lower bound is the maximum of this value and D. Denominators count
only retained original frame indices. Empty sensitivity windows have zero
counts and undefined percentages rather than invented zero-population evidence.
The selected replicas, original lengths and window settings are recorded.

Each starting model is summarized separately. The primary pooled scope combines
its requested replica files, weighting each saved frame equally, not each file
or starting model equally. Replica 0 is secondary if requested. Each selected
replica also has its own summary. Starts are not automatically pooled into a
single protein population, and no common time interval is silently imposed
when trajectories have different saved lengths.

## Joint structural clustering

The purpose is to describe sampled conformations beyond the reference states.
It does not replace direct RMSD comparisons or prove reference-state transitions.

By default every retained analysis frame enters the common clustering model
(`--cluster-max-frames 0`). Clustering and UMAP are enabled by default to restore
the full historical analysis suite. `--no-cluster` and `--no-umap` are explicit
opt-outs. A positive frame budget samples uniformly over eligible original
slot/frame indices, with a per-start share of the global budget. Short starts
can contribute fewer points. The scope (pooled or slot 0), fractional/absolute
prefixes, stride, per-start counts and every chosen frame are recorded.

Features combine:

1. Up to 512 variable nonlocal Cα soft contacts selected from at most 4,096
   deterministic residue pairs with ordinal gap ≥3. Contact value is
   `1 / (1 + (distance / 0.8 nm)^6)`. Variance in balanced samples capped at
   1,000 frames per start selects features independently of RMSD classifications;
   the selected contact features are then calculated over all clustered frames.
2. Sine/cosine encoding of backbone phi and psi. Each feature family is scaled
   by the square root of its feature count, without per-feature z-scoring.

PCA retains at least 90% of feature variance. HDBSCAN acts in that retained
space with leaf selection, default `min_samples=10` and
`min_cluster_size=max(40, ceil(0.01*sample_size))`. Public scikit-learn APIs are
used; no private hierarchy recutting or dense distance-matrix cache is imported
from the old study. Independent fits at 0.5%, 1%, and 2% cluster fractions
report group counts, noise and adjusted Rand agreement with the selected main fit.

Noise is label -1: no retained density-group assignment at the chosen resolution.
It remains in denominators and can contain sparse but meaningful conformations.
Membership strength is not the probability that a frame belongs to a biological
state. Optional seeded UMAP is a display coordinate only; labels come from
HDBSCAN in PCA space. PCA plots are used if UMAP is not requested.

Representatives are actual sampled frames nearest a cluster's PCA-space
centroid. They are full-atom source frames, exported with replica provenance.
`cluster_assignments.csv` and `cluster_populations.csv` provide the main labels
and per-start pooled/slot-0 fractions; metadata explicitly records whether every
retained analysis frame was assigned. Compatibility tables named `cluster_sample_*`
refer to the actual input to clustering even in full-frame mode. When a cap or
different clustering window is chosen, unsampled frames are not assigned.
Source-slot association diagnostics, group/noise populations, sensitivity plots
and full-atom representatives accompany the labels. These diagnose statistical
structure, not kinetic metastability or independent physical transitions.

## What a population claim requires

The default report measures structural sampling under a MELD protocol. It does
not establish unbiased physical probabilities or a converged rank order.
Different-temperature/alpha pooled replicas measure coverage, not one target
ensemble. Even replica 0 is only a candidate target-state ensemble: verify the
extractor indexes thermodynamic state rather than walker, the temperature and
restraint schedule, removal of startup/adaptation and stationary sampling.

Before combining starts to estimate a common restrained-ensemble distribution:

1. Establish that the same Hamiltonian/restraint model and target thermodynamic
   state apply. Different chemistry or confidence settings are separate conditions.
2. Determine a justified equilibrium window from actual saved run metadata;
   frame number is not necessarily cycle number. The default zero discard is
   intentionally an untrimmed coverage view.
3. Compare per-start and time-block assignments, effective sampling and repeated
   visits. Runs that retain starting-condition dependence should not be pooled
   to claim an equilibrium ranking.
4. Define state overlap/unassigned treatment, assess autocorrelation and
   independent repeats, and estimate uncertainty before interpreting differences.

The software does not implement a convergence proof, autocorrelation-corrected
confidence intervals, reweighting, free energies, tICA, MSMs, physical transition
rates, domain decomposition, or a final combined equilibrium ranking. These
should not be inferred from plots or small differences in sampled percentages.

## Reporting and the restored outputs

Every reference pair receives pooled and slot-0 backbone RMSD scatter plots,
with one panel per starting condition, matched axes, actual reference positions
and cutoff lines. Per-start plots include RMSD distributions, slot-0 frame
traces, per-slot reference occupancy and cutoff/prefix sensitivity. Pair-specific
A-only/B-only/both/neither accounting supplements multi-reference exclusive,
ambiguous and unassigned labels. A third reference can match a frame labeled
neither for a particular pair.

For every start/reference the analysis saves selected-frame CSVs and, by default,
full-atom cutoff DCDs aligned to that reference. Closest full-atom PDB/JSON exports
record actual frame provenance even when no cutoff hit exists. `--no-extract`
suppresses cutoff DCDs but keeps aligned totals, CSVs and closest snapshots.

The standalone HTML contains methods, reference-pair comparisons and conclusions,
per-start diagnostics, cluster diagnostics when present, cross-run conclusions
and embedded summary downloads. `RESULTS.md` supplies a short readable overview.
These are data-driven reports rather than the old study's fixed narrative.
An explicit report-only command can regenerate plots from CSVs, with an HTML
backup, but cannot create missing aligned trajectories, features or clustering.

## Reproducibility, safety and validation

The component reads only the input simulation tree. All summaries, figures,
manifests and optional representatives go to a fresh analysis directory. It
never creates Data/Logs, changes restraints, runs `launch_remd_multiplex`, submits
jobs, installs packages automatically, or edits source trajectories.

PDB/config/contact hashes, trajectory size/mtime signatures (or full hashes with
`--hash-dcd`), package versions and analysis source hashes are saved. Signatures
are rechecked at completion to detect concurrent modifications. Size/mtime
alone cannot detect every possible content edit; full hashes cost extra I/O.
The analysis assumes extraction is finished before reading. Failed/interrupted
outputs must not be mistaken for completed results. Reruns use a fresh directory.

Synthetic tests exercise arbitrary model names/counts, real DCD parsing,
alignment invariance and reflection rejection, overlap, frame accounting,
validation failures, no-write preflight, sample clustering, complete reporting
and full-atom aligned exports. No MD, scheduler or biological example is required.

## Origin and migration from the historical study

This workflow carries forward the scientific definitions from the old paired
study: independent matched backbone fits, explicit overlap/unassigned states,
4 Å main/2–5 Å sensitivity, pooled coverage and replica-0 reporting, internal
feature clustering and offline reports. The historical study itself remains in
the workspace's `meld_analysis/`, documented by its own `ANALYSIS.md`.

The integrated scripts replace fixed `set*/CoDNaS/meld_runs/state` discovery,
two-reference assumptions, fixed 30-replica checks, merged-frame offsets, VMD
requirements, private cached hierarchy calls and hand-reviewed study-specific
HTML assertions. They use the preparation manifest, N references, configurable
replicas, streamed numeric DCDs, public clustering APIs and data-driven reporting.
Old system-specific code and historical results are not copied into this package.
Consequently, the portable component does not promise identical historical
cluster labels or a copy of the historical narrative. Default full-frame
clustering and the optional all-frame independent validator provide the
corresponding scientific analyses without fixed protein identities or VMD.
