"""Bounded structural clustering on a documented subset, independent of RMSD labels."""
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
    ca = trajectory.top.select("name CA")
    i, j = np.triu_indices(len(ca), k=3)
    pairs = np.column_stack([ca[i], ca[j]])
    if len(pairs) > 4096:
        pairs = pairs[np.linspace(0, len(pairs)-1, 4096, dtype=int)]
    soft = 1 / (1 + (md.compute_distances(trajectory, pairs, periodic=False) / .8) ** 6) if len(pairs) else np.empty((len(xyz), 0))
    selected = np.argsort(-np.var(soft, axis=0), kind="stable")[:512]
    contacts = soft[:, selected]
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
    pd.DataFrame(sensitivity).to_csv(output / "clustering_sensitivity.csv", index=False)
    counts = []
    for name, group in provenance.groupby("start_model", sort=False):
        for label in sorted(set(labels)):
            count = int(np.sum(group.cluster == label))
            counts.append({"start_model": name, "cluster": int(label), "count": count,
                           "sampled_frames": len(group), "sample_pct": 100*count/len(group)})
    pd.DataFrame(counts).to_csv(output / "cluster_sample_populations.csv", index=False)
    reps = []
    representative_dir = output / "cluster_representatives"
    representative_dir.mkdir()
    for label in sorted(set(labels)-{-1}):
        members = np.flatnonzero(labels == label)
        centroid = reduced[members].mean(axis=0)
        pick = members[np.argmin(np.linalg.norm(reduced[members]-centroid, axis=1))]
        row = provenance.iloc[pick]
        run = next(r for r in runs if r["id"] == row.start_model)
        structure = md.load_frame(str(Path(run["directory"]) / f"{int(row.replica)}.dcd"),
                                  int(row.replica_frame), top=run["pdb"])
        filename = f"cluster_{label}.pdb"
        structure.save_pdb(str(representative_dir / filename))
        reps.append({"cluster": int(label), "start_model": row.start_model, "replica": int(row.replica),
                     "replica_frame": int(row.replica_frame), "pdb": f"cluster_representatives/{filename}"})
    pd.DataFrame(reps, columns=["cluster", "start_model", "replica", "replica_frame", "pdb"]).to_csv(
        output / "cluster_representatives.csv", index=False)
    dump_json(output / "clustering_metadata.json", {
        "scope": args.cluster_scope, "discard_fraction_per_replica": args.cluster_discard_fraction,
        "sampled_frames": len(xyz), "sample_counts_per_start": provenance.start_model.value_counts().to_dict(),
        "full_trajectory_assignments": False, "n_features": features.shape[1],
        "feature_contact_residue_ordinals_zero_based": [[int(trajectory.top.atom(int(a)).residue.index),
                                                       int(trajectory.top.atom(int(b)).residue.index)]
                                                      for a, b in pairs[selected]],
        "contact_switch_nm": .8, "contact_exponent": 6, "minimum_contact_sequence_gap": 3,
        "PCA_components": reduced.shape[1], "PCA_variance_retained": float(pca.explained_variance_ratio_.sum()),
        "cluster_selection_method": "leaf", "min_cluster_size": minimum,
        "min_samples": args.cluster_min_samples, "noise_label": -1,
        "n_clusters": len(set(labels)-{-1}), "noise_pct": 100*float(np.mean(labels == -1)),
        "UMAP_for_visualization_only": args.umap, "random_seed": 42})
