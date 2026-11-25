"""Data-driven portable report. No dataset-specific prose or expected result counts."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_report(output, manifest):
    runs = manifest["runs"]
    main = manifest["settings"]["main_cutoff"]
    primary = pd.read_csv(output / "primary_pooled_summary.csv", dtype={"start_model": str, "reference_model": str})
    partitions = pd.read_csv(output / "assignment_populations.csv", dtype={"start_model": str})
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
             "<title>MELD structural sampling analysis</title><style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:0 1em;color:#172334}img{max-width:100%}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:.4em;border:1px solid #ccd}p{line-height:1.5}code{background:#eef}</style>",
             "<h1>MELD structural sampling analysis</h1>",
             f"<p>{len(runs)} starting models; main backbone RMSD cutoff {main:g} Å. "
             "Primary: pooled numbered replicas. Secondary: replica 0.</p>",
             "<h2>Methods and interpretation</h2><p>N, Cα, C and O atoms are matched by residue ordinal, residue identity and atom name. "
             "Each frame is independently superposed on each prepared input with float64 Kabsch fitting and a proper rotation. "
             "Distances are unweighted RMSDs in Å; OXT and hydrogens are excluded. No coordinates are changed in the source files.</p>",
             "<p>Inclusive reference fractions can overlap and need not sum to 100%. Exclusive assignments, ambiguous (multiple references within cutoff), "
             "and unassigned (none within cutoff) partition every retained frame. Numbered DCD frame 0 is a real extracted sample and is retained unless "
             "excluded by the configured prefix. total.dcd is not read.</p>",
             "<p>Pooled fractions measure sampling coverage over the replica ladder, not equilibrium populations at a single temperature. "
             "Replica-0 fractions require verified extraction semantics, stationary temperature/restraints and equilibration before a population interpretation. "
             "Prefix sensitivity does not prove equilibration. Replica-exchange paths do not establish physical transition rates. "
             "No cross-start equilibrium distribution is inferred automatically.</p>",
             f"<p>Absolute prefix omitted per replica: {manifest['settings']['discard_frames']} frames; stride: {manifest['settings']['stride']}. "
             "Sensitivity fractions are applied per original replica length. Full settings and provenance are available in the downloads.</p>"]
    reference_map = pd.DataFrame([{"reference_id": r["reference_id"], "model": r["id"]} for r in runs])
    parts += ["<h2>Reference mapping and separation</h2>", reference_map.to_html(index=False, escape=True),
              pd.read_csv(output / "reference_rmsd.csv").to_html(index=False, escape=True)]
    images = []
    for scope, filename in [("pooled", "primary_pooled_summary.csv"), ("replica_0", "secondary_replica0_summary.csv")]:
        table = pd.read_csv(output / filename, dtype={"start_model": str, "reference_model": str})
        if table.empty:
            continue
        matrix = table.pivot(index="start_model", columns="reference_model", values="within_pct").reindex(
            index=[r["id"] for r in runs], columns=[r["id"] for r in runs])
        fig, ax = plt.subplots(figsize=(max(6, len(runs)*.55), max(4, len(runs)*.45)), constrained_layout=True)
        image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(runs)), [r["reference_id"] for r in runs], rotation=90)
        ax.set_yticks(range(len(runs)), [r["reference_id"] for r in runs])
        ax.set_xlabel("Reference conformation"); ax.set_ylabel("Starting model")
        ax.set_title(f"{scope}: inclusive matches at {main:g} Å")
        fig.colorbar(image, ax=ax, label="Retained sampled frames (%)")
        path = output / f"{scope}_reference_matches.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        images.append(path)
    parts.append("<h2>Comparisons by starting model</h2>")
    for run in runs:
        rows = primary[primary.start_model == run["id"]]
        unique = partitions[(partitions.start_model == run["id"]) & (partitions.scope == "pooled")
                            & (partitions.cutoff_A == main) & (partitions.discard_fraction_per_replica == 0)]
        parts += [f"<h3>{html.escape(run['id'])}</h3>", rows.to_html(index=False, escape=True),
                  "<p>Exclusive assignment accounting:</p>", unique.to_html(index=False, escape=True)]
        others = rows[rows.reference_model != run["id"]]
        detected = others[others.within_count > 0]
        parts.append(f"<p>{len(detected)} of {len(others)} other reference neighborhoods have at least one retained match. "
                     "Check exclusive counts and reference separation before interpreting those matches as distinct conformations.</p>")
    if (output / "cluster_sample_assignments.csv").exists():
        clusters = pd.read_csv(output / "cluster_sample_assignments.csv", dtype={"start_model": str})
        metadata = json.loads((output / "clustering_metadata.json").read_text())
        parts += ["<h2>Structural clustering of a bounded sample</h2>",
                  "<p>Periodic backbone torsions and variable Cα soft contacts form internal features. PCA retains at least 90% of their variance; "
                  "HDBSCAN leaf selection identifies density groups in that PCA space. Noise (-1) is unassigned density, not a declaration of unfolding. "
                  "Cluster fractions describe only the documented, capped sample. Unsampled frames are not assigned. UMAP, when requested, is visualization only.</p>",
                  f"<p>{metadata['sampled_frames']} sampled frames; {metadata['n_clusters']} clusters; {metadata['noise_pct']:.2f}% noise.</p>",
                  pd.read_csv(output / "cluster_sample_populations.csv").to_html(index=False, escape=True)]
        axes_names = ("UMAP1", "UMAP2") if "UMAP1" in clusters else ("PC1", "PC2")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for run in runs:
            points = clusters[clusters.start_model == run["id"]]
            axes[0].scatter(points[axes_names[0]], points[axes_names[1]], s=4, alpha=.5, label=run["reference_id"])
        axes[0].legend(); axes[0].set_title("Starting condition")
        for label, points in clusters.groupby("cluster"):
            color = "#999999" if label == -1 else plt.get_cmap("tab20")(int(label) % 20)
            axes[1].scatter(points[axes_names[0]], points[axes_names[1]], s=4, alpha=.5, color=color)
        axes[1].set_title("Density groups; gray = noise")
        for ax in axes:
            ax.set_xlabel(axes_names[0]); ax.set_ylabel(axes_names[1])
        path = output / "clustering.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        images.append(path)
    parts += ["<h2>Comparison across runs and conclusions</h2>",
              "<p>Compare rows across starting models and inspect replica-0 and discard sensitivities. Different reference occupancy fractions across starts "
              "can reflect incomplete sampling. Similar fractions alone do not prove convergence. Report overlap and unassigned frames explicitly; "
              "a high unassigned fraction calls for inspecting new conformations, domain motion, or structural deterioration. "
              "Use independent repeats and uncertainty analysis before claiming a converged population ranking.</p>"]
    for path in images:
        encoded = base64.b64encode(path.read_bytes()).decode()
        parts.append(f"<img alt='{html.escape(path.stem)}' src='data:image/png;base64,{encoded}'>")
    parts.append("<h2>Embedded data downloads</h2><p>This HTML includes its figures and the downloads below and can be shared by itself. "
                 "Full per-frame CSVs and source DCDs remain in the analysis/source directories.</p><ul>")
    downloads = sorted(output.glob("*.csv")) + [output / "numerical_audit.json"]
    for path in downloads:
        data = base64.b64encode(path.read_bytes()).decode()
        parts.append(f"<li><a download='{html.escape(path.name)}' href='data:application/octet-stream;base64,{data}'>{html.escape(path.name)}</a></li>")
    # Embed a completed view; the on-disk manifest is finalized after this build.
    provenance = dict(manifest, status="complete")
    data = base64.b64encode(json.dumps(provenance, indent=2).encode()).decode()
    parts.append(f"<li><a download='analysis_manifest.json' href='data:application/json;base64,{data}'>analysis_manifest.json</a></li></ul></html>")
    (output / "report.html").write_text("\n".join(parts), encoding="utf-8")
