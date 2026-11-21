# Automated MELD pipeline

Put raw AF2 PDB models of **one protein** in `structures/`. The pipeline prepares
one independent simulation per model, with the same core and alternative contact
restraints in every simulation. Any positive number of models is supported;
different proteins/sequences need separate pipeline outputs.

This implements the normal-MELD restrained-population protocol developed in this
project. The master script only
prepares files. A separate command submits jobs to Slurm after you edit the HPC settings.

This working copy already has all 13 supplied models prepared in `simulations/`.
Use `python pipeline.py --resume` to verify that example. For changed inputs or a
new protein, choose a fresh output with `--output new_simulations`.

## 1. Environment installation

On an HPC, first load the site's conda/mamba module. Names vary; use `module avail`
and replace the example below with the appropriate module. On a workstation with
conda already installed, omit the module command.

```bash
module load mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n meld_new
conda activate meld_new
conda install -c conda-forge openmm openmpi mpi4py meld biopython mdtraj mdanalysis ambertools
command -v pdb4amber
command -v tleap
```

AmberTools provides `pdb4amber` and `tleap`. Preparation needs these tools but does
not need a GPU. Python 3.10 or newer is required by these scripts.

### CUDA compiler for running MELD

The validated MELD 0.6.1/OpenMM 7.7 environment requires an actual `nvcc` compiler,
in addition to CUDA runtime libraries. On HPC, load the site's matching toolkit:

```bash
module avail cuda
module load nvidia/cuda
which nvcc
nvcc --version
export OPENMM_CUDA_COMPILER="$(command -v nvcc)"
```

The CUDA module above is an example: use your HPC's full module name/version,
compatible with its driver and the CUDA runtime in your MELD environment. Export
the compiler path in the same job/shell that launches MELD. Do not assume
`/usr/local/cuda/bin/nvcc` exists.

## 2. Prepare your models

Copy `.pdb` files into `structures/`, including subfolders if useful. Every PDB is
included; remove unwanted relaxed/unrelaxed duplicates yourself. Source files are
preserved. Filenames containing spaces and repeated basenames in subfolders are
handled using unique numbered simulation-directory names.

```bash
conda activate meld_new
cd /path/to/Automated_MELD_pipeline
python pipeline.py
```

For each model, the master script:

1. Copies the raw model into its own folder under `simulations/`.
2. Runs `pdb4amber -i ../raw_af2.pdb -o pdb4amber.pdb` inside its `preparation/` folder.
3. Runs `tleap -f tleap.in` there, then copies `prot_meld_input.pdb` to the simulation folder.
4. Generates the common core/alternative contacts from all prepared models, using
   pLDDT from the original AF2 files and checking sequence correspondence.
5. Writes `sequence.dat`, `sequence.fasta`, and `residue_map.csv` from each prepared PDB.
6. Copies the setup/launch scripts, identical contact files, `input.json`, and a
   sample `job.slurm` into every simulation folder, together with a trajectory-extraction helper.

No MELD dynamics or job submission occurs during this command. Data/Logs are
normally created later by the Slurm job's setup step.

Other locations or confidence settings:

```bash
python pipeline.py --structures /path/to/models --output /path/to/new_simulations
python pipeline.py --output simulations_70 --core-confidence 0.95 --alternative-confidence 0.70
```

An existing output is protected. To resume an interrupted preparation or verify a
complete, unchanged preparation:

```bash
python pipeline.py --resume
```

Resume checks source and staged checksums. A changed model set, contact definition,
or MELD input requires a fresh output directory. Editing Slurm job files is allowed.

## 3. Configure and submit Slurm jobs

### A. Find your HPC settings

A Slurm job file tells the scheduler which resources to reserve and which commands
to run. The supplied `templates/job.slurm` is a sample, not a ready-to-submit file:
its `YOUR_...` values must be replaced with settings from your HPC.

Do this on the HPC login node. Have your project's allocation/account name and
the site's GPU-job instructions available. These commands can help you inspect
partitions and installed software modules:

```bash
sinfo
module avail
```

`sinfo` lists partitions; use your HPC documentation to identify the GPU partition
you are allowed to use. The allocation/account name comes from your HPC onboarding
information or project administrator; it is not necessarily your login username.
Module names and versions also come from your site's documentation or module list.

### B. Edit the central job template

From the pipeline's top-level folder, open the template in your preferred editor:

```bash
cd /path/to/Automated_MELD_pipeline
nano templates/job.slurm
```

In nano, save with **Ctrl+O**, press **Enter**, and exit with **Ctrl+X**.

Use this table while editing:

| Template setting | What to enter or check |
|---|---|
| `--account=YOUR_ACCOUNT` | Your project/allocation account; remove this directive if your site does not require it. |
| `--partition=YOUR_GPU_PARTITION` | Your permitted GPU partition. |
| `--nodes=1` | One node for this example. |
| `--ntasks=1` | One task for the multiplex launcher. |
| `--cpus-per-task=8` | Example CPU request; adjust for your site and workload. |
| `--gres=gpu:1` | One GPU; replace with your site's GPU-request syntax if necessary. |
| `--mem=32G` | Example memory per node. |
| `--time=24:00:00` | Example wall-time limit; stay within partition limits. |
| `--output` / `--error` | Log filenames; `%x` is the job name and `%j` the job ID. |
| `module load YOUR_CONDA_MODULE` | The site's conda/mamba module name. |
| `module load YOUR_CUDA_MODULE` | The compatible CUDA toolkit module name/version. |
| `conda activate meld_new` | The environment you installed; update if you chose another name. |

The CPU, memory, and time values are examples, not measured requirements for every
protein. Some sites use `--gpus=1` or a GPU type in `--gres`; follow their example
and replace the GPU directive rather than adding conflicting requests.

Keep `__JOB_NAME__` in the central template: the pipeline fills it automatically
with a distinct name when copying the job to each model folder. Keep all `#SBATCH`
directives above executable commands. Slurm does not expand shell variables inside
these directives. [Slurm directive reference](https://slurm.schedmd.com/sbatch.html)

If conda is already provided by your login environment, you may remove its module
line. If the compiler is installed in your conda environment, the CUDA module line
may also be unnecessary. Remove unused placeholder lines instead of leaving
`YOUR_...` values behind. Keep the environment activation and compiler checks.

Before submission, test your chosen module names and environment interactively on
the login node, using the corresponding commands from your edited template:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate meld_new
which python
which nvcc
nvcc --version
```

Load the modules first if your site requires them. `which nvcc` must report an
executable path; the job exports that path as `OPENMM_CUDA_COMPILER`. Finding the
compiler is a prerequisite, not a test that GPU execution has succeeded. A login
node need not have a GPU; run MELD only inside the appropriate GPU allocation.

You can check the edited file without executing its commands:

```bash
bash -n templates/job.slurm
grep -n 'YOUR_' templates/job.slurm
```

No output from `bash -n` means the shell syntax passed. No output from `grep`
means no `YOUR_` placeholders remain. Valid syntax alone does not verify your
allocation, partition access, or resource availability.

### C. Copy the edited template and preview

Edit the central template once, then copy it to every model and preview:

```bash
python submit_jobs.py --template templates/job.slurm --dry-run
```

This command **copies the template even in dry-run mode**, backs up changed
existing job files, and checks the generated files. It submits zero jobs. Open one
generated `simulations/model_.../job.slurm` and confirm the job name and HPC settings.
If you edit the central template again, repeat this copy/preview command so the
simulation folders receive the changes.

Alternatively, edit each model's existing `job.slurm` separately. In that case,
omit `--template` to retain those individual edits:

```bash
python submit_jobs.py --dry-run
```

### D. Submit and check the queue

After the preview is satisfactory, submit the existing per-model job files:

```bash
python submit_jobs.py --submit
```

Run this from the pipeline's top-level folder on the HPC, with `meld_new` active.
The helper queues one job per model. It does not guarantee that all jobs start
together; Slurm decides when the requested resources are available. A successful
submission supplies a job ID, not a completed simulation.

**Only `--submit` calls `sbatch`.** Submission is blocked if any pending job still
contains placeholders, has a Bash syntax error, or has changed/missing staged
inputs. The helper invokes `sbatch --parsable job.slurm` from each model's folder,
so `SLURM_SUBMIT_DIR` points to the correct simulation directory.

Successful job IDs are appended to `simulations/submissions.jsonl`. Already
submitted models are skipped on subsequent calls. After checking previous jobs
are no longer active and choosing the appropriate continuation workflow, an
intentional resubmission can use `--resubmit --submit`.

Check your jobs with:

```bash
squeue -u "$USER"
```

`PD` means pending and `R` means running. Once a job starts, inspect its
`slurm-*.out`, `slurm-*.err`, and `Logs/remd.log` inside the model folder. A job
disappearing from the queue does not establish success; check its logs.
[Queue command reference](https://slurm.schedmd.com/squeue.html)

Scheduler reference: [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html).

Use `--simulations /path/to/other_output` with either submission mode for a custom
output. Transfer this whole pipeline folder, including `templates/` and the
prepared output, to the HPC; no sibling project directories are required.

### What each Slurm job executes

The sample loads the configured modules, activates `meld_new`, finds `nvcc`, and
sets `OPENMM_CUDA_COMPILER`. It then:

```bash
if [[ ! -e Data && ! -e Logs ]]; then
    python system_setup.py
else
    python system_setup.py --check
fi
bash run_meld.sh
```

The launcher validates the setup and runs
`launch_remd_multiplex --platform CUDA --debug`. It multiplexes replicas within a
run; the example requests one GPU and one Slurm task, not one GPU per replica.
Independent model jobs still need separate allocations. For interrupted runs,
use the installed MELD version's restart workflow; rebuilding Data starts a new run.

## 4. Extract trajectories after the simulations

Two extraction scripts are provided:

- `templates/extract_trajectories.sh`: the replica-extraction loop for one simulation;
  the master also copies it into every model folder as `extract_trajectories.sh`.
- `extract_all_trajectories.py`: runs that helper in every model directory listed
  in the preparation manifest, sequentially.

Activate the MELD environment that can read the simulation output. From the
pipeline's top-level folder, preview and then extract all model trajectories:

```bash
conda activate meld_new
python extract_all_trajectories.py --dry-run
python extract_all_trajectories.py
```

For a custom output directory:

```bash
python extract_all_trajectories.py --simulations /path/to/new_simulations
```

For just one model, enter that model's simulation folder and run:

```bash
bash extract_trajectories.sh
```

The helper reads `n_replicas` from `input.json`, checking it against
`setup_report.json` if present. It executes the supplied MELD command for each
valid index, equivalent to:

```bash
for index in $(seq 0 1 29); do
    filename=$index.dcd
    echo "Running: extract_trajectory extract_traj_dcd --replica $index $filename"
    extract_trajectory extract_traj_dcd --replica "$index" "$filename"
done
```

**Replica indices start at zero.** The default 30 replicas are numbered **0-29**.
The originally suggested `seq 0 1 30` covers 31 replicas and is valid only for a
simulation actually configured with 31 replicas. The scripts determine the last
index automatically and reject out-of-range overrides.

Optional extraction settings, supported by both scripts:

```bash
python extract_all_trajectories.py --first-replica 0 --last-replica 9
python extract_all_trajectories.py --overwrite
```

Existing nonempty DCDs are skipped by default. New outputs are written to hidden
temporary DCDs and renamed to `0.dcd`, `1.dcd`, etc. only after a successful
extraction. `--overwrite` replaces existing outputs after successful extraction;
use it if an earlier output is incomplete. Nonempty files are not otherwise
checked for trajectory integrity.

Each model retains `extract_trajectories.log` with extraction output and errors.
The batch script writes `simulations/extraction_summary.json` with per-model exit
codes. It continues to the remaining models after a failed extraction and exits
nonzero if any model failed. Inspect that model's log and rerun when resolved;
completed nonempty DCDs will be skipped. A dry run writes no files and can preview
commands before Data exists. Actual extraction requires saved MELD Data and
`extract_trajectory` on PATH; it reads the frames available in the saved output.

Outputs remain inside each model folder alongside `prot_meld_input.pdb`. These
scripts extract separate replica trajectories; subsequent alignment, merging,
and structural-population analysis are separate processing steps.

## Files and editable protocol

```text
Automated_MELD_pipeline/
  pipeline.py                  # Master preparation entry point
  submit_jobs.py               # Job copying, preview, and explicit submission
  extract_all_trajectories.py   # Extract replicas from every model folder
  README.md
  structures/                  # User's raw AF2 PDBs
  templates/
    input.json                 # Editable protocol defaults
    tleap.in
    generate_contacts.py
    pdb_to_fasta.py
    structure_utils.py
    system_setup.py
    run_meld.sh
    extract_trajectories.sh
    job.slurm                  # Edit for the destination HPC
  simulations/
    pipeline_manifest.json
    contacts/
    reports/
    model_0001_.../
      raw_af2.pdb
      preparation/             # pdb4amber and tleap intermediate files
      preparation_logs/
      prot_meld_input.pdb
      sequence.dat
      sequence.fasta
      residue_map.csv
      contacts/
      input.json
      system_setup.py
      run_meld.sh
      extract_trajectories.sh
      job.slurm
```

Edit `templates/input.json` before preparing a new condition, or pass `--config`
to the master script. The per-model `input.json` records its actual staged values;
model identifiers, residue count, and a distinct Langevin seed are filled automatically.
Contact definitions live under `contact_generation`; enforcement fractions live
under `collections.core.confidence` and `collections.alternatives.confidence`.
Do not edit one staged model's restraint protocol and then combine its populations
with models using different settings. Prepare a separate output for each condition.

Defaults preserve the established protocol:

- C-alpha contact cutoff 8 A; sequence separation >=6.
- Core: present in all models, spread <=1 A, pLDDT >=80 at both residues in every
  raw AF2 model. Flat-bottom interval covers the observed range plus 0.5 A per side.
- Alternatives: remaining union contacts; upper zero-energy bound 8.5 A.
- Core confidence 0.95; alternative confidence 0.65, configurable (e.g. 0.60/0.70).
- Separate collections of singleton contacts; lowest-energy requested number
  active each step; nearest-integer rounding with halves upward.
- Peak spring constant 350 kJ/mol/nm^2; contact CSV bounds stored in **nm**.
- ff14sbside, GBNeck2, AMAP disabled, 2 fs timestep; 30 replicas over 300-400 K.
- 25,000 MD steps/cycle, 20,000 cycles (1 microsecond per replica if completed).
- Startup ramp through cycle 200; ladder adaptation through cycle 1000.
- Restraints active in the analysis replica, weakened up the ladder. No additional
  Cartesian, torsion, or longer-distance collection, and no MELD-Adapt.

Counts depend on the input models. See `reports/contact_summary.json` and
`reports/input_restraint_audit.csv` for exact counts, active subsets, and initial
penalties at 60%, 65%, and 70% alternatives confidence. Keeping confidence fixed
while adding conformations changes the restraint model. These audits are not
force-field energies or population estimates.

An empty core or alternative collection is recorded and omitted from MELD rather
than inventing contacts. Both empty is an error. With one model, all geometry is
shared and no multi-model state-ranking comparison is available. No spatial
alignment is needed to compute internal C-alpha distances.

## Scope, checks, and continuation

Inputs must be single-model, single-chain, standard-amino-acid protein PDBs with
matching full sequences. Multimers, ligands/HETATM records, unresolved alternative
locations, CYX/disulfide topology, and differing prepared protonation/chemistry
require explicit preparation beyond this standard pipeline. Unsupported inputs
stop with an explanation rather than being silently dropped. Original AF2 B-factors
provide confidence; zero confidence produces fewer/no trusted core contacts.

One model's setup can be tested without MD, from its individual simulation folder:

```bash
python system_setup.py
python system_setup.py --check
```

This creates and reads back `Data/` and `Logs/`, but performs no minimization or
integration and does not test CUDA kernel compilation. Setup never overwrites
existing Data/Logs. `--archive-existing` explicitly preserves an old setup before
building a fresh one; it is not a restart command. `setup_report.json` records
actual versions, hashes, settings, and validation.

Future population analysis must use equilibrated frames at the same temperature
and restraint conditions across starts, demonstrate convergence, and allow new or
intermediate states. Count structural states, not active restraint-group IDs.
Different-temperature pooled trajectories are a coverage overview, not direct
target-ensemble populations. State-assignment cutoffs must suit the particular
input conformations; the earlier kinase 00/08 pair is only 3.36 A apart.

Tests (AmberTools and the example PDB inputs are needed for integration checks):

```bash
python -m unittest discover -s tests -v
```

Tests use temporary outputs, a fake scheduler, and a fake trajectory extractor;
they do not launch MD, submit real HPC jobs, or extract actual trajectories.
The existing `kinase_meld_4states/` project is separate from this
new pipeline and is not used as a runtime dependency.
