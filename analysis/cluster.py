"""Joint full-frame or explicitly sampled structural clustering, independent of RMSD labels."""
import math
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from .metrics import dump_json


def cluster_sample(xyz, records, runs, output, args):
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score

    trajectory = md.Trajectory(np.asarray(xyz), runs[0]["backbone_reference"].top)
    provenance = pd.DataFrame(records)
    eligible_total = 0
    for run in runs:
        for item in run["files"]:
            start = args.discard_frames
            eligible_total += len(range(start, item["frames"], args.stride))
    full_assignments = len(records) == eligible_total
    ca = trajectory.top.select("name CA")
    i, j = np.triu_indices(len(ca), k=3)
    pairs = np.column_stack([ca[i], ca[j]])
    if len(pairs) > 4096:
        pairs = pairs[np.linspace(0, len(pairs)-1, 4096, dtype=int)]
    # Choose contact features using equally capped samples per start, as in the
    # historical analysis; compute only the selected features over all frames.
    balanced = []
    for _, group in provenance.groupby("start_model", sort=False):
        indices = group.index.to_numpy()
        balanced.extend(indices[np.linspace(0, len(indices)-1, min(1000, len(indices)), dtype=int)])
    candidate = md.compute_distances(trajectory[np.asarray(balanced)], pairs, periodic=False) if len(pairs) else np.empty((len(balanced), 0))
    soft = 1 / (1 + (candidate / .8) ** 6)
    selected = np.argsort(-np.var(soft, axis=0), kind="stable")[:512]
    contacts = 1 / (1 + (md.compute_distances(trajectory, pairs[selected], periodic=False) / .8) ** 6) if len(selected) else np.empty((len(xyz), 0))
    _, phi = md.compute_phi(trajectory, periodic=False)
    _, psi = md.compute_psi(trajectory, periodic=False)
    angles = np.concatenate([phi, psi], axis=1)
    torsions = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    features = np.concatenate([contacts / np.sqrt(max(1, contacts.shape[1])),
                               torsions / np.sqrt(max(1, torsions.shape[1]))], axis=1)
    if not features.shape[1] or not np.isfinite(features).all():
        raise ValueError("No finite internal structural features available for clustering.")
    if np.sum(np.var(features, axis=0)) < 1e-12:
        raise ValueError("Clustering sample has no structural variation; run RMSD-only analysis instead.")
    pca = PCA(n_components=.90, svd_solver="full")
    reduced = pca.fit_transform(features)
    minimum = max(args.cluster_min_size, math.ceil(len(xyz)*args.cluster_fraction))
    if minimum > len(xyz) or args.cluster_min_samples > len(xyz):
        raise ValueError("Clustering sample is too small for requested cluster size/min_samples.")

    def fit(size):
        return HDBSCAN(min_cluster_size=size, min_samples=args.cluster_min_samples,
                       cluster_selection_method="leaf", n_jobs=args.threads).fit(reduced)

    model = fit(minimum)
    labels = model.labels_
    sensitivity = []
    for fraction in sorted(set([.005, .01, .02, args.cluster_fraction])):
        size = max(args.cluster_min_size, math.ceil(len(xyz)*fraction))
        if size > len(xyz):
            continue
        other = labels if size == minimum else fit(size).labels_
        sensitivity.append({"fraction": fraction, "min_cluster_size": size,
                            "n_clusters": len(set(other)-{-1}), "noise_pct": 100*float(np.mean(other == -1)),
                            "ARI_to_main_including_noise": float(adjusted_rand_score(labels, other))})
    provenance["cluster"] = labels
    provenance["membership_strength"] = model.probabilities_
    provenance["PC1"] = reduced[:, 0]
    provenance["PC2"] = reduced[:, 1] if reduced.shape[1] > 1 else 0.
    if args.umap:
        import umap
        if len(xyz) < 5:
            raise ValueError("UMAP needs at least five sampled frames.")
        embedding = umap.UMAP(n_components=2, n_neighbors=min(30, len(xyz)-1), min_dist=.15,
                              n_epochs=200, random_state=42, n_jobs=1).fit_transform(reduced)
        provenance["UMAP1"], provenance["UMAP2"] = embedding.T
    provenance.to_csv(output / "cluster_sample_assignments.csv", index=False)
    provenance.to_csv(output / "cluster_assignments.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "clustering_sensitivity.csv", index=False)
    counts = []
    for name, group in provenance.groupby("start_model", sort=False):
        for label in sorted(set(labels)):
            count = int(np.sum(group.cluster == label))
            counts.append({"start_model": name, "cluster": int(label), "count": count,
                           "sampled_frames": len(group), "sample_pct": 100*count/len(group)})
    pd.DataFrame(counts).to_csv(output / "cluster_sample_populations.csv", index=False)
    populations = []
    for name, group in provenance.groupby("start_model", sort=False):
        for scope, sub in [("pooled", group), ("replica_0", group[group.replica == 0])]:
            for label in sorted(set(labels)):
                count = int(np.sum(sub.cluster == label))
                populations.append({"start_model": name, "scope": scope, "cluster": int(label),
                                    "count": count, "n_frames": len(sub),
                                    "pct": 100*count/len(sub) if len(sub) else None,
                                    "all_retained_frames_assigned": full_assignments})
    pd.DataFrame(populations).to_csv(output / "cluster_populations.csv", index=False)
    diagnostics = []
    for label in sorted(set(labels)-{-1}):
        group = provenance[provenance.cluster == label]
        sources = group.groupby(["start_model", "replica"]).size()
        diagnostics.append({"cluster": int(label), "n_frames": len(group), "distinct_source_slots": len(sources),
                            "distinct_starts": group.start_model.nunique(),
                            "dominant_source_slot_fraction": float(sources.max()/len(group))})
    pd.DataFrame(diagnostics, columns=["cluster", "n_frames", "distinct_source_slots", "distinct_starts", "dominant_source_slot_fraction"]).to_csv(
        output / "cluster_source_diagnostics.csv", index=False)
    source_codes = pd.factorize(pd.MultiIndex.from_frame(provenance[["start_model", "replica"]]))[0]
    source_ari = float(adjusted_rand_score(labels, source_codes))
    assigned = sum(d["n_frames"] for d in diagnostics)
    source_purity = sum(d["n_frames"]*d["dominant_source_slot_fraction"] for d in diagnostics)/assigned if assigned else None
    reps = []
    representative_dir = output / "cluster_representatives"
    representative_dir.mkdir()
    for label in sorted(set(labels)-{-1}):
        members = np.flatnonzero(labels == label)
        centroid = reduced[members].mean(axis=0)
        pick = members[np.argmin(np.linalg.norm(reduced[members]-centroid, axis=1))]
        row = provenance.iloc[pick]
        run = next(r for r in runs if r["id"] == row.start_model)
        structure = md.load_frame(str(output / run["reference_id"] / "total.dcd"),
                                  int(row.total_frame), top=run["reference"].top)
        filename = f"cluster_{label}.pdb"
        structure.save_pdb(str(representative_dir / filename))
        reps.append({"cluster": int(label), "start_model": row.start_model, "replica": int(row.replica),
                     "replica_frame": int(row.replica_frame), "total_frame": int(row.total_frame),
                     "pdb": f"cluster_representatives/{filename}"})
    pd.DataFrame(reps, columns=["cluster", "start_model", "replica", "replica_frame", "total_frame", "pdb"]).to_csv(
        output / "cluster_representatives.csv", index=False)
    dump_json(output / "clustering_metadata.json", {
        "scope": args.cluster_scope, "discard_fraction_per_replica": args.cluster_discard_fraction,
        "sampled_frames": len(xyz), "sample_counts_per_start": provenance.start_model.value_counts().to_dict(),
        "full_trajectory_assignments": full_assignments, "n_features": features.shape[1],
        "frame_budget": args.cluster_max_frames, "total_retained_analysis_frames": eligible_total,
        "feature_selection": "variance in capped balanced samples across starts; no RMSD labels",
        "feature_contact_residue_ordinals_zero_based": [[int(trajectory.top.atom(int(a)).residue.index),
                                                       int(trajectory.top.atom(int(b)).residue.index)]
                                                      for a, b in pairs[selected]],
        "contact_switch_nm": .8, "contact_exponent": 6, "minimum_contact_sequence_gap": 3,
        "PCA_components": reduced.shape[1], "PCA_variance_retained": float(pca.explained_variance_ratio_.sum()),
        "cluster_selection_method": "leaf", "min_cluster_size": minimum,
        "min_samples": args.cluster_min_samples, "noise_label": -1,
        "n_clusters": len(set(labels)-{-1}), "noise_pct": 100*float(np.mean(labels == -1)),
        "UMAP_for_visualization_only": args.umap, "random_seed": 42})
    pd.DataFrame([{"n_clusters": len(set(labels)-{-1}), "noise_pct": 100*float(np.mean(labels == -1)),
                   "PCA_components": reduced.shape[1], "PCA_variance_retained": float(pca.explained_variance_ratio_.sum()),
                   "weighted_dominant_source_slot_fraction": source_purity, "ARI_clusters_vs_source_slot": source_ari,
                   "full_trajectory_assignments": full_assignments}]).to_csv(output / "clustering_overview.csv", index=False)
