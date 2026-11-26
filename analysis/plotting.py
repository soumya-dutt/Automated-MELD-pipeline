"""RMSD diagnostics from saved CSVs; no trajectory or source-path access required."""
from itertools import combinations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


def read_csv(path):
    return pd.read_csv(path, dtype={"start_model": str, "reference_model": str})


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.stem + ".tmp.png")
    try:
        fig.savefig(temporary, dpi=130, format="png")
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return path


def short_name(run):
    return run["reference_id"].replace("reference_", "R")


def retained_window(table, run, settings, fraction=0.):
    lengths = {item["replica"]: item["frames"] for item in run["files"]}
    minimum = table.replica.map(lambda replica: max(settings["discard_frames"],
                                                    int(np.floor(lengths[replica]*fraction))))
    return table[table.replica_frame >= minimum]


def pair_membership(table, a, b, cutoff):
    close_a = table[a + "_rmsd_A"].to_numpy() <= cutoff
    close_b = table[b + "_rmsd_A"].to_numpy() <= cutoff
    return {"A_only": int(np.sum(close_a & ~close_b)),
            "B_only": int(np.sum(close_b & ~close_a)),
            "both": int(np.sum(close_a & close_b)),
            "neither": int(np.sum(~close_a & ~close_b))}


def generate_diagnostics(output, manifest, reference_pairs=None, discard_fraction=0.):
    """Use every retained frame in the chosen window; all reference pairs by default."""
    output = Path(output)
    runs, settings = manifest["runs"], manifest["settings"]
    cutoff = settings["main_cutoff"]
    ids = [run["reference_id"] for run in runs]
    columns = [identifier + "_rmsd_A" for identifier in ids]
    tables = {}
    for run in runs:
        table = read_csv(output / run["reference_id"] / "frame_rmsd.csv")
        if table.empty or not np.isfinite(table[columns].to_numpy()).all():
            raise ValueError(f"Missing/invalid frame RMSDs: {run['reference_id']}")
        if not (table.start_model == run["id"]).all():
            raise ValueError(f"Frame table belongs to another start: {run['reference_id']}")
        tables[run["reference_id"]] = retained_window(table, run, settings, discard_fraction)
    populations = read_csv(output / "reference_populations.csv")
    partitions = read_csv(output / "assignment_populations.csv")
    reference_table = read_csv(output / "reference_rmsd.csv")
    separations = {(row.reference_a, row.reference_b): row.rmsd_A
                   for row in reference_table.itertuples()}
    pairs = list(combinations(ids, 2)) if reference_pairs is None else list(reference_pairs)
    for a, b in pairs:
        if a not in ids or b not in ids or a == b:
            raise ValueError(f"Unknown/repeated reference in pair: {a}:{b}")
    result = {"per_start": {}, "pairs": [], "global": [], "tables": tables,
              "pair_count": len(pairs), "discard_fraction": discard_fraction}
    statistics, closest, counts = [], [], []
    colors = {identifier: plt.get_cmap("tab20")(i % 20) for i, identifier in enumerate(ids)}
    for run in runs:
        identifier = run["reference_id"]
        table = tables[identifier]
        folder = output / identifier
        plots = []
        # Matched distributions have the same histogram bins and axes across scopes.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        upper = max([cutoff * 1.1] + [float(table[c].max()) for c in columns if len(table)])
        bins = np.linspace(0, upper, 55)
        for ax, scope, subset in [(axes[0], "Pooled slots", table),
                                   (axes[1], "Slot 0", table[table.replica == 0])]:
            for ref in runs:
                column = ref["reference_id"] + "_rmsd_A"
                if len(subset):
                    ax.hist(subset[column], bins=bins, density=True, histtype="step", lw=1.5,
                            color=colors[ref["reference_id"]], label=short_name(ref))
            ax.axvline(cutoff, ls="--", color="#c62828", label=f"{cutoff:g} Å cutoff")
            ax.set(xlabel="Backbone RMSD to reference (Å)", ylabel="Probability density (Å⁻¹)",
                   title=f"{scope}; n={len(subset):,}", xlim=(0, upper))
            ax.legend(fontsize=7)
        fig.suptitle(f"Start {short_name(run)}: RMSD distributions")
        plots.append(("Backbone RMSD distributions", save_figure(fig, folder / "rmsd_distributions.png")))

        fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
        slot0 = table[table.replica == 0].sort_values("replica_frame")
        for ref in runs:
            ax.plot(slot0.replica_frame, slot0[ref["reference_id"] + "_rmsd_A"], lw=.8,
                    color=colors[ref["reference_id"]], label=short_name(ref))
        ax.axhline(cutoff, color="#c62828", ls="--")
        ax.set(xlabel="Original frame index within extracted slot-0 DCD",
               ylabel="Backbone RMSD (Å)", title=f"Start {short_name(run)}: slot 0; n={len(slot0):,}")
        ax.legend(fontsize=7, ncol=min(4, len(runs)))
        plots.append(("Slot-0 RMSD traces", save_figure(fig, folder / "replica_0_rmsd_trace.png")))

        own = populations[(populations.start_model == run["id"]) & (populations.cutoff_A == cutoff)
                          & np.isclose(populations.discard_fraction_per_replica, discard_fraction)]
        slots = own[own.scope.str.fullmatch(r"replica_\d+")].copy()
        slots["slot"] = slots.scope.str.rsplit("_", n=1).str[-1].astype(int)
        matrix = slots.pivot(index="reference_id", columns="slot", values="within_pct").reindex(index=ids)
        if not matrix.empty:
            fig, ax = plt.subplots(figsize=(max(7, .23*len(matrix.columns)), max(2.8, .4*len(runs))), constrained_layout=True)
            image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="viridis")
            ticks = np.arange(0, len(matrix.columns), max(1, len(matrix.columns)//15))
            ax.set_xticks(ticks, matrix.columns.to_numpy()[ticks])
            ax.set_yticks(range(len(runs)), [short_name(r) for r in runs])
            ax.set(xlabel="Temperature/alpha slot index", ylabel="Reference", title=f"Start {short_name(run)}: inclusive matches at {cutoff:g} Å")
            fig.colorbar(image, ax=ax, label="Retained frames (%)")
            plots.append(("Reference matches by temperature/alpha slot", save_figure(fig, folder / "reference_matches_by_replica.png")))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, scope in zip(axes, ["pooled", "replica_0"]):
            for ref in runs:
                sub = populations[(populations.start_model == run["id"]) & (populations.scope == scope)
                                  & (populations.reference_id == ref["reference_id"])
                                  & np.isclose(populations.discard_fraction_per_replica, discard_fraction)].sort_values("cutoff_A")
                ax.plot(sub.cutoff_A, sub.within_pct, "o-", ms=3,
                        color=colors[ref["reference_id"]], label=short_name(ref))
            ax.axvline(cutoff, ls="--", color="#c62828", lw=.8)
            ax.set(xlabel="Backbone RMSD cutoff (Å)", ylabel="Inclusive matching frames (%)", ylim=(0, 100), title=scope)
            ax.legend(fontsize=7)
        fig.suptitle(f"Start {short_name(run)}: cutoff sensitivity")
        plots.append(("RMSD cutoff sensitivity", save_figure(fig, folder / "cutoff_sensitivity.png")))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, scope in zip(axes, ["pooled", "replica_0"]):
            for ref in runs:
                sub = populations[(populations.start_model == run["id"]) & (populations.scope == scope)
                                  & (populations.reference_id == ref["reference_id"])
                                  & (populations.cutoff_A == cutoff)].sort_values("discard_fraction_per_replica")
                ax.plot(100*sub.discard_fraction_per_replica, sub.within_pct, "o-", ms=3,
                        color=colors[ref["reference_id"]], label=short_name(ref))
            ax.set(xlabel="Fractional prefix excluded per slot (%)", ylabel="Inclusive matching frames (%)", ylim=(0, 100), title=scope)
            ax.legend(fontsize=7)
        fig.suptitle(f"Start {short_name(run)}: prefix sensitivity at {cutoff:g} Å")
        plots.append(("Discarded-prefix sensitivity", save_figure(fig, folder / "discard_sensitivity.png")))
        result["per_start"][identifier] = plots

        for scope, sub in [("pooled", table), ("replica_0", slot0)]:
            for ref in runs:
                values = sub[ref["reference_id"] + "_rmsd_A"]
                statistics.append({"start_model": run["id"], "scope": scope, "reference_id": ref["reference_id"],
                                   "reference_model": ref["id"], "n_frames": len(sub), "cutoff_A": cutoff,
                                   "discard_fraction_per_replica": discard_fraction,
                                   "minimum_rmsd_A": values.min(), "median_rmsd_A": values.median(),
                                   "p05_rmsd_A": values.quantile(.05), "p95_rmsd_A": values.quantile(.95)})
                if len(sub):
                    pick = sub.loc[values.idxmin()]
                    closest.append({"start_model": run["id"], "scope": scope,
                                    "reference_id": ref["reference_id"], "reference_model": ref["id"],
                                    "discard_fraction_per_replica": discard_fraction,
                                    "replica": int(pick.replica), "replica_frame": int(pick.replica_frame),
                                    "rmsd_A": float(values.min()), "within_main_cutoff": bool(values.min() <= cutoff)})

    for a, b in pairs:
        separation = float(separations[(a, b)])
        plots = []
        bound = max([cutoff*1.1, separation*1.1] + [float(t[c].max())*1.02 for t in tables.values()
                                                   for c in [a+"_rmsd_A", b+"_rmsd_A"] if len(t)])
        for scope in ["pooled", "replica_0"]:
            ncols = min(3, len(runs))
            nrows = math.ceil(len(runs)/ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.3*ncols, 4.1*nrows),
                                     squeeze=False, constrained_layout=True)
            max_color = max([1] + [int(t.replica.max()) if scope == "pooled" else int(t.replica_frame.max())
                                  for t in tables.values() if len(t)])
            norm = Normalize(0, max_color)
            image = None
            for ax, run in zip(axes.flat, runs):
                table = tables[run["reference_id"]]
                sub = table if scope == "pooled" else table[table.replica == 0]
                groups = pair_membership(sub, a, b, cutoff)
                counts.append({"reference_a": a, "reference_b": b, "reference_separation_A": separation,
                               "cutoff_neighborhoods_can_overlap": separation <= 2*cutoff,
                               "start_model": run["id"], "scope": scope, "cutoff_A": cutoff,
                               "discard_fraction_per_replica": discard_fraction, "n_frames": len(sub), **groups})
                if len(sub):
                    image = ax.scatter(sub[a+"_rmsd_A"], sub[b+"_rmsd_A"],
                                       c=sub.replica if scope == "pooled" else sub.replica_frame,
                                       cmap="viridis", norm=norm, s=5 if scope == "pooled" else 14,
                                       alpha=.35 if scope == "pooled" else .75, linewidths=0)
                ax.plot([0, bound], [0, bound], color="#aaaaaa", lw=.6)
                ax.axvline(cutoff, color="#c62828", ls="--", lw=.9)
                ax.axhline(cutoff, color="#c62828", ls="--", lw=.9)
                ax.scatter([0, separation], [separation, 0], marker="*", c="#ffb300", edgecolors="black", s=95, zorder=5)
                ax.set(xlim=(0, bound), ylim=(0, bound), aspect="equal",
                       xlabel=f"RMSD to {a.replace('reference_', 'R')} (Å)",
                       ylabel=f"RMSD to {b.replace('reference_', 'R')} (Å)",
                       title=f"Start {short_name(run)}; n={len(sub):,}")
            for ax in list(axes.flat)[len(runs):]:
                ax.set_visible(False)
            if image is not None:
                # Use an opaque colorbar even when plotted points are transparent.
                from matplotlib.cm import ScalarMappable
                fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes.flat)[:len(runs)],
                             label="Temperature/alpha slot index" if scope == "pooled" else "Original slot-0 frame index",
                             shrink=.8)
            fig.suptitle(f"{a.replace('reference_', 'R')} vs {b.replace('reference_', 'R')} — {scope}\n"
                         f"References {separation:.3f} Å apart; dashed lines = {cutoff:g} Å cutoff; stars = references")
            destination = output / "pairwise" / f"{a}__{b}" / f"rmsd_scatter_{scope}.png"
            plots.append((scope, save_figure(fig, destination)))
        # Retain the original study's log-count hexbin view, with pooled and
        # slot-0 rows. Paginate starts so arbitrary model counts stay readable.
        for page_start in range(0, len(runs), 4):
            page = runs[page_start:page_start+4]
            fig, axes = plt.subplots(2, len(page), figsize=(4.1*len(page), 7.6), squeeze=False, constrained_layout=True)
            for col, run in enumerate(page):
                table = tables[run["reference_id"]]
                for row, (scope, sub) in enumerate([("pooled", table), ("replica_0", table[table.replica == 0])]):
                    ax = axes[row, col]
                    if len(sub):
                        image = ax.hexbin(sub[a+"_rmsd_A"], sub[b+"_rmsd_A"], gridsize=40,
                                          extent=(0, bound, 0, bound), mincnt=1, bins="log", cmap="viridis")
                        fig.colorbar(image, ax=ax, label="Frame count (log scale)", shrink=.8)
                    ax.axvline(cutoff, color="#c62828", ls="--", lw=.9)
                    ax.axhline(cutoff, color="#c62828", ls="--", lw=.9)
                    ax.scatter([0, separation], [separation, 0], marker="*", c="#ffb300", edgecolors="black", s=75)
                    ax.set(xlim=(0, bound), ylim=(0, bound), aspect="equal",
                           xlabel=f"RMSD to {a.replace('reference_', 'R')} (Å)",
                           ylabel=f"RMSD to {b.replace('reference_', 'R')} (Å)",
                           title=f"Start {short_name(run)} — {scope}; n={len(sub):,}")
            fig.suptitle(f"Two-reference RMSD density; separation {separation:.3f} Å")
            suffix = "" if page_start == 0 else f"_page_{page_start//4+1}"
            destination = output / "pairwise" / f"{a}__{b}" / f"two_reference_rmsd{suffix}.png"
            plots.append((f"pooled and replica_0 density, page {page_start//4+1}", save_figure(fig, destination)))
        result["pairs"].append({"a": a, "b": b, "separation_A": separation, "plots": plots})
    pd.DataFrame(statistics).to_csv(output / "rmsd_statistics.csv", index=False)
    pd.DataFrame(closest).to_csv(output / "closest_reference_frames.csv", index=False)
    fields = ["reference_a", "reference_b", "reference_separation_A", "cutoff_neighborhoods_can_overlap",
              "start_model", "scope", "cutoff_A", "discard_fraction_per_replica", "n_frames", "A_only", "B_only", "both", "neither"]
    pd.DataFrame(counts, columns=fields).to_csv(output / "pairwise_reference_populations.csv", index=False)
    # Exclusive categories form an honest denominator, including all unmatched frames.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = ids + ["ambiguous", "unassigned"]
    for ax, scope in zip(axes, ["pooled", "replica_0"]):
        base = np.zeros(len(runs))
        for label in labels:
            sub = partitions[(partitions.scope == scope) & (partitions.cutoff_A == cutoff)
                             & np.isclose(partitions.discard_fraction_per_replica, discard_fraction)
                             & (partitions.assignment == label)].set_index("start_model")
            values = sub.pct.reindex([r["id"] for r in runs]).fillna(0).to_numpy()
            color = colors.get(label, "#bdbdbd" if label == "ambiguous" else "#eeeeee")
            ax.bar(range(len(runs)), values, bottom=base, color=color, edgecolor="#666666", lw=.3,
                   label=label.replace("reference_", "R"))
            base += values
        ax.set_xticks(range(len(runs)), [short_name(r) for r in runs], rotation=45)
        ax.set(ylim=(0, 100), xlabel="Starting model", ylabel="Retained sampled frames (%)", title=scope)
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    result["global"].append(("Exclusive multi-reference assignments", save_figure(fig, output / "assignment_populations.png")))
    return result


def cluster_diagnostics(output, runs, frame_tables):
    """UMAP/PCA views, full label heatmaps, stacked occupancies and source checks."""
    output = Path(output)
    clusters = read_csv(output / "cluster_sample_assignments.csv")
    populations = read_csv(output / "cluster_populations.csv") if (output / "cluster_populations.csv").exists() else None
    axes_names = ("UMAP1", "UMAP2") if "UMAP1" in clusters else ("PC1", "PC2")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for run in runs:
        points = clusters[clusters.start_model == run["id"]]
        axes[0].scatter(points[axes_names[0]], points[axes_names[1]], s=3, alpha=.5, label=short_name(run))
    axes[0].legend(fontsize=7); axes[0].set_title("Starting condition")
    for label, points in clusters.groupby("cluster"):
        color = "#999999" if label == -1 else plt.get_cmap("tab20")(int(label) % 20)
        axes[1].scatter(points[axes_names[0]], points[axes_names[1]], s=3, alpha=.5, color=color)
    axes[1].set_title("HDBSCAN groups; gray = noise")
    minimum = np.full(len(clusters), np.nan)
    for run in runs:
        mask = clusters.start_model == run["id"]
        frame = frame_tables[run["reference_id"]]
        index = frame.set_index(["replica", "replica_frame"])
        subset = clusters[mask]
        keys = pd.MultiIndex.from_arrays([subset.replica, subset.replica_frame])
        columns = [r["reference_id"] + "_rmsd_A" for r in runs]
        # A report window may exclude frames that were used in clustering.
        minimum[mask] = index[columns].min(axis=1).reindex(keys).to_numpy()
    image = axes[2].scatter(clusters[axes_names[0]], clusters[axes_names[1]], c=minimum, cmap="viridis", s=3, alpha=.6)
    fig.colorbar(image, ax=axes[2], label="Minimum RMSD to any reference (Å)")
    axes[2].set_title("Reference similarity; not a state assignment")
    for ax in axes:
        ax.set_xlabel(axes_names[0]); ax.set_ylabel(axes_names[1])
    plots = [("Structural clustering and reference similarity", save_figure(fig, output / "structural_clusters_umap.png"))]
    if populations is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, scope in zip(axes, ["pooled", "replica_0"]):
            sub = populations[populations.scope == scope]
            matrix = sub.pivot(index="cluster", columns="start_model", values="pct").reindex(columns=[r["id"] for r in runs])
            image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="viridis")
            ax.set_xticks(range(len(runs)), [short_name(r) for r in runs], rotation=45)
            step = max(1, len(matrix)//25)
            positions = list(range(0, len(matrix), step))
            ax.set_yticks(positions, ["noise" if matrix.index[i] == -1 else str(matrix.index[i]) for i in positions])
            ax.set(title=scope, xlabel="Starting model", ylabel="Cluster label (local to this protein)")
            fig.colorbar(image, ax=ax, label="Clustered-scope frames (%)")
        plots.append(("All cluster populations", save_figure(fig, output / "cluster_population_heatmap.png")))
        sizes = clusters[clusters.cluster != -1].groupby("cluster").size().sort_values(ascending=False)
        top = sizes.head(10).index.tolist()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for ax, scope in zip(axes, ["pooled", "replica_0"]):
            bottom = np.zeros(len(runs))
            sub = populations[populations.scope == scope]
            for position, label in enumerate(top + ["remaining", "noise"]):
                if label == "remaining": selected = sub[(sub.cluster != -1) & ~sub.cluster.isin(top)]
                elif label == "noise": selected = sub[sub.cluster == -1]
                else: selected = sub[sub.cluster == label]
                heights = selected.groupby("start_model").pct.sum().reindex([r["id"] for r in runs]).fillna(0).to_numpy()
                color = "#bbbbbb" if label == "noise" else "#dddddd" if label == "remaining" else plt.get_cmap("tab20")(position % 20)
                ax.bar(range(len(runs)), heights, bottom=bottom, color=color, label=str(label))
                bottom += heights
            ax.set_xticks(range(len(runs)), [short_name(r) for r in runs], rotation=45)
            ax.set(title=scope, ylim=(0, 100), xlabel="Starting model", ylabel="Clustered-scope frames (%)")
        axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        plots.append(("Top groups, remaining groups and noise", save_figure(fig, output / "cluster_populations.png")))
    sensitivity = read_csv(output / "clustering_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    axes[0].plot(sensitivity["fraction"]*100, sensitivity.n_clusters, "o-")
    axes[0].set(xlabel="Minimum cluster fraction (%)", ylabel="Number of clusters")
    axes[1].plot(sensitivity["fraction"]*100, sensitivity.noise_pct, "o-")
    axes[1].set(xlabel="Minimum cluster fraction (%)", ylabel="Noise (%)", ylim=(0, 100))
    plots.append(("HDBSCAN resolution sensitivity", save_figure(fig, output / "clustering_sensitivity.png")))
    return plots
