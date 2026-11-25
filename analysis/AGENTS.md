# Agent execution guide

## Scope

This package performs postprocessing only. Read `README.md` for commands and
`ANALYSIS.md` for scientific definitions. The top-level `analyze_trajectories.py`
is the entry point. No external workspace directory is a runtime dependency.

## Reliable execution sequence

1. Locate the user's complete preparation manifest and unchanged, extracted
   numeric DCDs. One manifest represents one protein sequence and one protocol.
2. Use the analysis environment. Run `python analyze_trajectories.py --help`
   and a `--dry-run` with explicit input/output paths.
3. Resolve missing replicas or sequence/protocol inconsistencies from evidence;
   do not silently omit files or change chemistry. An explicit `--replicas`
   option is available when the requested scope is genuinely a subset.
4. Use a new output directory and record the chosen cutoffs, windows and stride.
   Defaults produce a coverage survey. Do not invent a burn-in-to-cycle conversion
   or label a survey as an equilibrium population ranking.
5. Add `--cluster` only when structural clustering is requested/useful and its
   bounded sample fits available resources. UMAP is optional display only.
6. Check manifest status `complete`, numerical audit `passed`, output row counts
   and exclusive/ambiguous/unassigned accounting. A `running` or `failed` manifest
   is an incomplete result even if some figures exist.
   Run `python -m analysis.validate_results --analysis OUTPUT` to audit saved
   accounting. Use `--full-rmsd` for an explicit independent all-frame check when
   the extra trajectory I/O is appropriate; verify `validation.json: status=passed`.
7. Report the output path, scope/window, reference mapping, overlap/unassigned
   fractions and material limitations. The HTML is shareable alone; raw/frame-level
   data and representative coordinates require their companion files.

Do not run simulations, submit jobs, modify trajectories or install packages as
a side effect of an analysis command. Never repair source data by changing its
manifest to hide a failed check. Never write special cases for protein IDs or
expected match counts. Keep script defaults relative to the project/CLI, and
keep all thresholds configurable. Do not merge starts into a thermodynamic
population without a separately justified analysis.

## Changes and tests

Use synthetic, temporary fixtures for tests. From repository root:

```bash
python -m unittest discover -s tests -p test_analysis.py -v
```

Test structural correspondence and proper rotations, original replica/frame
indices, multi-reference overlap, missing/corrupt input behavior and unchanged
source files for any relevant code change. Optional clustering percentages
refer to sampled frames only. If the execution contract changes, update both
this package's documentation and the top-level README analysis section.
