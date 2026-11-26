"""Standalone reports and full RMSD diagnostics; can read saved CSVs offline."""
import argparse
import base64
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "meld-analysis-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import digest, dump_json
from .plotting import generate_diagnostics, read_csv, save_figure, short_name


def build_report(output, manifest, reference_pairs=None, discard_fraction=0.):
    output = Path(output)
    runs, settings = manifest["runs"], manifest["settings"]
    if not runs or len({r["reference_id"] for r in runs}) != len(runs):
        raise ValueError("A nonempty list of unique reference IDs is required.")
    if any(not r["reference_id"].startswith("reference_") or not r["reference_id"][10:].isdigit() for r in runs):
        raise ValueError("Invalid reference ID in manifest.")
    if not np.isfinite(discard_fraction) or not any(np.isclose(discard_fraction, f) for f in settings["discard_fractions"]):
        raise ValueError("Choose a discard fraction present in the saved settings.")
    main = settings["main_cutoff"]
    source_paths = [output / name for name in ("reference_populations.csv",
                   "assignment_populations.csv", "reference_rmsd.csv", "numerical_audit.json")]
    source_paths += [output / r["reference_id"] / "frame_rmsd.csv" for r in runs]
    source_hashes = {str(path.relative_to(output)): digest(path) for path in source_paths}
    diagnostics = generate_diagnostics(output, manifest, reference_pairs, discard_fraction)
    populations = read_csv(output / "reference_populations.csv")
    partitions = read_csv(output / "assignment_populations.csv")
    selected = populations[(populations.cutoff_A == main) & np.isclose(populations.discard_fraction_per_replica, discard_fraction)]
    accounting = partitions[(partitions.cutoff_A == main) & np.isclose(partitions.discard_fraction_per_replica, discard_fraction)]
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
             "<title>MELD structural sampling analysis</title><style>body{font:16px system-ui;max-width:1350px;margin:2em auto;padding:0 1em;color:#172334}img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:.4em;border:1px solid #ccd}p,li{line-height:1.5}details{border:1px solid #ccd;padding:.7em;margin:1em 0}summary{cursor:pointer;font-weight:bold}.note{background:#fff3d6;padding:1em}</style>",
             "<h1>MELD structural sampling analysis</h1>",
             f"<p>{len(runs)} starting models; cutoff {main:g} Å. Pooled slots are primary coverage; slot 0 is secondary.</p>",
             f"<p>Displayed fractional exclusion: {100*discard_fraction:g}% per slot; absolute prefix: {settings['discard_frames']} frames; stride: {settings['stride']}.</p>",
             "<nav><a href='#methods'>Methods</a> · <a href='#overview'>Overview</a> · <a href='#pairs'>Reference pairs</a> · <a href='#starts'>Starting models</a> · <a href='#clustering'>Clustering</a> · <a href='#conclusions'>Conclusions</a> · <a href='#downloads'>Downloads</a></nav>",
             "<h2 id='methods'>Methods and interpretation</h2>",
             "<p>Full-atom trajectories are fitted on N/Cα/C/O onto each start's prepared input before analysis and saved as reference_*/total.dcd with an explicit original slot/frame map. All selected slots and frames enter this merged DCD, including frames subsequently excluded or strided for analysis. No synthetic topology frame is added. Each RMSD comparison then independently fits the matched backbone onto its chosen reference with float64 Kabsch proper rotations. Values are in Å; OXT and hydrogens are excluded from fitting.</p>" if manifest.get("alignment") else "<p>Each frame is independently fitted on matched N/Cα/C/O atoms onto each reference using float64 Kabsch proper rotations. Values are in Å. This older result does not include the new aligned total.dcd output.</p>",
             "<p>Inclusive reference fractions can overlap. Exclusive reference labels, ambiguous (multiple references within cutoff) and unassigned (none within cutoff) partition every frame. Pair-specific A-only/B-only/both/neither labels describe only the displayed two references; pairwise neither can include matches to a third reference.</p>",
             "<p>The standard MELD extract_traj_dcd command selects a temperature/alpha slot. Slot 0 is the lowest-temperature slot, not one walker followed through exchanges. Pooled frames describe coverage across temperatures/restraint strengths, not one equilibrium ensemble. Prefix sensitivity does not establish equilibration, and replica exchange does not supply ordinary physical transition kinetics.</p>",
             "<p>Scatter plots use every retained frame in the displayed window, with common axes for each reference pair. Stars mark input references and dashed lines mark the cutoff. Color is slot index for pooled frames and original DCD frame index for slot 0. Trace x axes are saved frame indices, not physical walker time.</p>"]
    images = []

    def figure(path, caption):
        path = Path(path); images.append(path)
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"<figure><img alt='{html.escape(caption, quote=True)}' src='data:image/png;base64,{encoded}'><figcaption>{html.escape(caption)}</figcaption></figure>"

    def table(data):
        return data.to_html(index=False, escape=True, float_format=lambda v: f"{v:.4g}")

    parts += ["<h3>Reference mapping</h3>", table(pd.DataFrame([{"plot_label": short_name(r), "reference_id": r["reference_id"], "model": r["id"]} for r in runs]))]
    validation = output / "validation.json"
    if validation.exists():
        parts += ["<h3>Saved validation</h3><pre>" + html.escape(validation.read_text()) + "</pre>"]
    parts.append("<h2 id='overview'>Across-start sampling overview</h2>")
    overview = []
    for scope in ["pooled", "replica_0"]:
        sub = selected[selected.scope == scope]
        if sub.empty:
            continue
        matrix = sub.pivot(index="start_model", columns="reference_model", values="within_pct").reindex(index=[r["id"] for r in runs], columns=[r["id"] for r in runs])
        fig, ax = plt.subplots(figsize=(max(6, len(runs)*.55), max(4, len(runs)*.45)), constrained_layout=True)
        image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(runs)), [short_name(r) for r in runs], rotation=90)
        ax.set_yticks(range(len(runs)), [short_name(r) for r in runs])
        ax.set(xlabel="Reference conformation", ylabel="Starting model", title=f"{scope}: inclusive matches at {main:g} Å")
        fig.colorbar(image, ax=ax, label="Retained sampled frames (%)")
        parts.append(figure(save_figure(fig, output / f"{scope}_reference_matches.png"), f"Inclusive occupancy: {scope}"))
        for run in runs:
            row = sub[sub.start_model == run["id"]]
            if row.empty:
                continue
            counts = accounting[(accounting.start_model == run["id"]) & (accounting.scope == scope)].set_index("assignment")
            overview.append({"start": short_name(run), "scope": scope, "n_frames": int(row.n_frames.iloc[0]),
                             "within_own_reference_pct": float(row[row.reference_id == run["reference_id"]].within_pct.iloc[0]),
                             "ambiguous_pct": float(counts.loc["ambiguous", "pct"]), "unassigned_pct": float(counts.loc["unassigned", "pct"])})
    for caption, path in diagnostics["global"]:
        parts.append(figure(path, caption))
    overview_table = pd.DataFrame(overview)
    overview_table.to_csv(output / "sampling_overview.csv", index=False)
    parts += [table(overview_table), "<details><summary>Reference separation matrix entries</summary>", table(read_csv(output / "reference_rmsd.csv")), "</details>"]
    parts.append("<h2 id='pairs'>Reference-pair comparisons and pair-level conclusions</h2>")
    pair_populations = read_csv(output / "pairwise_reference_populations.csv")
    if not diagnostics["pairs"]:
        parts.append("<p>One reference is available; no between-reference scatter comparison is possible.</p>")
    for pair in diagnostics["pairs"]:
        a, b = pair["a"], pair["b"]
        parts.append(f"<h3>{html.escape(a)} versus {html.escape(b)}</h3><p>Reference separation: {pair['separation_A']:.3f} Å. " +
                     ("Cutoff neighborhoods can overlap; interpret inclusive matches with both-reference and exclusive counts." if pair["separation_A"] <= 2*main else "Cutoff neighborhoods are separated by the reference-distance criterion.") + "</p>")
        for scope, path in pair["plots"]:
            parts.append(figure(path, f"Backbone RMSD scatter: {scope}; {a} versus {b}"))
        rows = pair_populations[(pair_populations.reference_a == a) & (pair_populations.reference_b == b)].copy()
        rows["start"] = rows.start_model.map({r["id"]: short_name(r) for r in runs})
        parts.append(table(rows[["start", "scope", "n_frames", "A_only", "B_only", "both", "neither"]]))
        counts = rows[rows.scope == "pooled"]
        parts.append(f"<p>Across the displayed starts, {int(counts.both.sum()):,} pooled frames match both references and {int(counts.neither.sum()):,} match neither. These aggregate counts describe sampling, not a combined equilibrium population.</p>")
    parts.append("<h2 id='starts'>Per-start comparisons and diagnostics</h2>")
    stats = read_csv(output / "rmsd_statistics.csv")
    for run in runs:
        parts += [f"<h3>Start {short_name(run)}: {html.escape(run['id'])}</h3>", table(selected[selected.start_model == run["id"]])]
        for caption, path in diagnostics["per_start"][run["reference_id"]]:
            parts.append(figure(path, caption))
        parts += ["<details><summary>Minimum, median and percentile RMSDs</summary>", table(stats[stats.start_model == run["id"]]), "</details>"]
    parts.append("<h2 id='clustering'>Structural clustering</h2>")
    if (output / "cluster_sample_assignments.csv").exists():
        metadata = json.loads((output / "clustering_metadata.json").read_text())
        complete = metadata.get("full_trajectory_assignments", False)
        parts.append("<p>Periodic backbone torsions and variable nonlocal Cα soft contacts form the features. PCA retains at least 90% variance; HDBSCAN leaf selection clusters PCA space. Noise (-1) remains in population denominators. UMAP is visualization only. Representatives are actual frames nearest each cluster's PCA-space centroid.</p>")
        parts.append(f"<p>All retained trajectory frames assigned: {complete}. Clustered frames: {metadata['sampled_frames']:,}; groups: {metadata['n_clusters']}; noise: {metadata['noise_pct']:.2f}%. "
                     "Clustering uses its separately recorded scope/window. A report rebuild does not recluster.</p>")
        parts.append("<details><summary>Structural feature and clustering settings</summary><pre>" + html.escape(json.dumps(metadata, indent=2)) + "</pre></details>")
        if (output / "clustering_overview.csv").exists():
            parts.append(table(read_csv(output / "clustering_overview.csv")))
            parts.append("<p>Source-slot association describes how density groups relate to starting condition and temperature/alpha slot. Association can reflect different thermodynamic conditions; it does not establish kinetic states or prove an artifact.</p>")
        from .plotting import cluster_diagnostics
        for caption, path in cluster_diagnostics(output, runs, diagnostics["tables"]):
            parts.append(figure(path, caption))
        parts.append(table(read_csv(output / "clustering_sensitivity.csv")))
    else:
        parts.append("<p class='note'>Clustering was not run in this result. Its internal contact/torsion features cannot be recovered from RMSD tables. Run the full analysis where the PDB/DCD files are available to obtain HDBSCAN, UMAP and representatives.</p>")
    parts += ["<h2 id='conclusions'>Comparison across all runs and conclusions</h2>", table(overview_table),
              "<p>Compare starts, slot 0, and cutoff/prefix sensitivities. Persistent starting-condition differences can reflect incomplete sampling. Similar percentages alone do not establish convergence. Report overlapping and unassigned frames explicitly. A high unassigned fraction calls for inspecting new conformations, domain motion or structural deterioration; global RMSD alone cannot distinguish them. Independent repeats and uncertainty analysis are needed for a converged population ranking.</p>"]
    markdown = ["# MELD structural sampling results", "", f"Main cutoff {main:g} Å. Pooled slots are primary coverage; slot 0 is secondary.",
                f"Absolute exclusion {settings['discard_frames']} frames; fractional exclusion {discard_fraction:g}; stride {settings['stride']}.", "",
                "| Start | Scope | Frames | Within own reference | Ambiguous | Unassigned |", "|---|---|---:|---:|---:|---:|"]
    markdown += [f"| {r['start']} | {r['scope']} | {r['n_frames']} | {r['within_own_reference_pct']:.3f}% | {r['ambiguous_pct']:.3f}% | {r['unassigned_pct']:.3f}% |" for r in overview]
    markdown += ["", "Reference mapping:", ""] + [f"- {short_name(r)}: {r['id']}" for r in runs]
    markdown += ["", "See report.html for pairwise pooled/slot-0 scatter plots, distributions, traces, slot occupancy, sensitivities and clustering diagnostics. Pooled fractions are not a single-temperature equilibrium population. Pair-specific neither can include matches to other references."]
    (output / "RESULTS.md").write_text("\n".join(markdown) + "\n")
    parts.append("<h2 id='downloads'>Embedded data downloads</h2><p>Every displayed figure and the downloads below, including available closest/reference/cluster PDBs, are embedded for offline sharing. Full trajectories and per-frame tables remain companion files.</p><ul>")
    downloads = sorted(output.glob("*.csv")) + [output / "numerical_audit.json", output / "RESULTS.md"]
    if (output / "clustering_metadata.json").exists(): downloads.append(output / "clustering_metadata.json")
    if validation.exists(): downloads.append(validation)
    downloads += sorted(output.glob("reference_*/prot_meld_input.pdb"))
    downloads += sorted(output.glob("reference_*/closest_to_reference_*.pdb"))
    downloads += sorted(output.glob("reference_*/closest_to_reference_*.json"))
    downloads += sorted((output / "cluster_representatives").glob("*.pdb"))
    for path in downloads:
        data = base64.b64encode(path.read_bytes()).decode()
        label = str(path.relative_to(output))
        filename = label.replace("/", "_")
        parts.append(f"<li><a download='{html.escape(filename, quote=True)}' href='data:application/octet-stream;base64,{data}'>{html.escape(label)}</a></li>")
    data = base64.b64encode(json.dumps(dict(manifest, status="complete"), indent=2).encode()).decode()
    parts.append(f"<li><a download='analysis_manifest.json' href='data:application/json;base64,{data}'>analysis_manifest.json</a></li>")
    metadata = {"schema_version": 2, "status": "complete", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "discard_fraction_per_replica": discard_fraction, "reference_pairs": [[p["a"], p["b"]] for p in diagnostics["pairs"]],
                "embedded_figures": len(images), "input_csv_json_sha256": source_hashes,
                "plot_code_sha256": {name: digest(Path(__file__).parent / name) for name in ("report.py", "plotting.py")},
                "trajectories_read": False, "RMSDs_recomputed": False, "clustering_recomputed": False}
    if any(digest(output / name) != expected for name, expected in source_hashes.items()):
        raise ValueError("An analysis input changed during report generation.")
    data = base64.b64encode(json.dumps(metadata, indent=2).encode()).decode()
    parts.append(f"<li><a download='report_manifest.json' href='data:application/json;base64,{data}'>report_manifest.json</a></li></ul></html>")
    temporary = output / ".report.html.tmp"
    temporary.write_text("\n".join(parts), encoding="utf-8")
    temporary.replace(output / "report.html")
    dump_json(output / "report_manifest.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path(__file__).resolve().parents[1] / "analysis_results")
    parser.add_argument("--discard-fraction", type=float, default=0.)
    parser.add_argument("--reference-pairs", nargs="+", help="Optional reference_0000:reference_0001 pairs; default all pairs")
    args = parser.parse_args(argv)
    try:
        output = args.analysis.resolve()
        manifest = json.loads((output / "analysis_manifest.json").read_text())
        if manifest.get("status") != "complete": raise ValueError("A completed analysis is required.")
        pairs = None if args.reference_pairs is None else [tuple(s.split(":")) for s in args.reference_pairs]
        if pairs is not None and any(len(p) != 2 for p in pairs): raise ValueError("Use reference_0000:reference_0001 pair syntax.")
        backup = None
        if (output / "report.html").exists():
            backup = output / "report_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            backup.mkdir(parents=True)
            shutil.copy2(output / "report.html", backup / "report.html")
        metadata = build_report(output, manifest, pairs, args.discard_fraction)
        print(json.dumps({"report": str(output / "report.html"), "previous_report_backup": str(backup) if backup else None,
                          "embedded_figures": metadata["embedded_figures"], "trajectories_read": False}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
