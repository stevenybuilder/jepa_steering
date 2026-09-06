"""TRAIN-fitted local charts for dense patch geometry; no manifold assumption.

Use one block and imagined horizon per call. Rows are action alternatives, not
independent episodes. The global, local and random-neighborhood charts have the
same rank; all hyperparameters are fixed before reading development outcomes.
"""
from __future__ import annotations
import numpy as np
import torch


def gram_pca(values, rank):
    if values.ndim != 2 or len(values) < 3 or not torch.isfinite(values).all():
        raise ValueError("Finite matrix with at least3 samples required")
    x = values.float()
    mean = x.mean(0)
    centered = x-mean
    # Sample-space eigensolve avoids a102400-by102400 covariance matrix.
    gram = centered@centered.T
    eigenvalues, left = torch.linalg.eigh(gram.double())
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, left = eigenvalues[order], left[:, order]
    keep = (eigenvalues > max(float(eigenvalues[0])*1e-7, 1e-12)).nonzero().flatten()[:rank]
    if not len(keep):
        raise ValueError("No identifiable nonzero chart dimension")
    basis = (left[:, keep].T@centered.double())/eigenvalues[keep].sqrt()[:, None]
    return mean, basis.float(), (eigenvalues[keep]/(len(values)-1)).float()


def balanced_neighbors(distances, groups, count, max_per_group=4):
    if len(distances) != len(groups) or count < 3:
        raise ValueError("Invalid neighborhood dimensions")
    selected, counts = [], {}
    for index in torch.argsort(distances).cpu().tolist():
        group = str(groups[index])
        if counts.get(group, 0) >= max_per_group:
            continue
        selected.append(index); counts[group] = counts.get(group, 0)+1
        if len(selected) == count:
            return selected
    raise ValueError("Too few independent groups for balanced fixed neighborhood")


def subspace_overlap(first, second):
    if first.shape[1] != second.shape[1]:
        raise ValueError("Subspaces need the same ambient coordinates")
    cosines = torch.linalg.svdvals(first.double()@second.double().T).clamp(0, 1)
    return {"principal_cosines": cosines.cpu().tolist(),
            "mean_squared_principal_cosine": float(cosines.square().mean())}


def distribution_summary(scores, groups):
    """Descriptive tails/dependence, not Gaussianity proof or native beliefs."""
    from scipy.stats import chi2
    x = scores.double().cpu().numpy()
    if x.ndim != 2 or len(x) != len(groups):
        raise ValueError("Scores/group axes differ")
    radius = np.sum(x*x, axis=1)
    threshold = float(chi2.ppf(.95, x.shape[1]))
    variance = np.mean((x-x.mean(0))**2, axis=0)
    standardized = (x-x.mean(0))/np.sqrt(np.maximum(variance, 1e-12))
    kurtosis = np.mean(standardized**4, axis=0)-3
    square = x*x
    if x.shape[1] > 1 and np.all(square.std(0) > 1e-9):
        corr = np.corrcoef(square, rowvar=False)
        dependence = float(np.mean(np.abs(corr[np.triu_indices(x.shape[1], 1)])))
    else:
        dependence = None
    rows = []
    for group in sorted(set(str(g) for g in groups)):
        take = np.asarray([str(g) == group for g in groups])
        rows.append({"episode": group, "rows": int(take.sum()),
                     "mean_squared_radius": float(radius[take].mean()),
                     "gaussian_reference_tail_fraction": float((radius[take] > threshold).mean())})
    return {"coordinate_excess_kurtosis": kurtosis.tolist(), "squared_score_dependence": dependence,
            "gaussian_reference_radius95": threshold, "episode_rows": rows,
            "experimental_units": len(rows), "gaussian_assumed_for_reference_only": True,
            "native_model_probabilities": False}


def ridge_prediction(train_x, train_y, test_x, alpha=100.):
    xmean, ymean = train_x.mean(0), train_y.mean(0)
    centered = train_x-xmean
    eye = torch.eye(centered.shape[1], device=centered.device, dtype=centered.dtype)
    weights = torch.linalg.solve(centered.T@centered+alpha*eye, centered.T@(train_y-ymean))
    return (test_x-xmean)@weights+ymean


@torch.no_grad()
def analyze_chart(train, test, train_groups, test_groups, *, rank=8, ambient_rank=32,
                  neighbors=32, max_per_group=4, train_targets=None, test_targets=None):
    if set(map(str, train_groups)) & set(map(str, test_groups)):
        raise ValueError("Discovery/development episode overlap refused")
    if len(train) != len(train_groups) or len(test) != len(test_groups):
        raise ValueError("Sample/group axes differ")
    if (train_targets is None) != (test_targets is None):
        raise ValueError("Both target arrays or neither required")
    train, test = train.float(), test.float()
    mean, ambient, variance = gram_pca(train, ambient_rank)
    if len(ambient) < rank or neighbors <= rank:
        raise ValueError("Insufficient support for requested equal chart rank")
    tr = (train-mean)@ambient.T
    te = (test-mean)@ambient.T
    # Same TRAIN-fitted ambient and raw metric for every chart. No test refit.
    train_total = float((train-mean).square().sum(1).mean())
    omitted = ((test-mean).square().sum(1)-te.square().sum(1)).clamp_min(0)
    global_basis = torch.eye(len(ambient), device=train.device)[:rank]
    global_error = omitted+(te-te@global_basis.T@global_basis).square().sum(1)
    scaled_train = tr/variance.sqrt().clamp_min(1e-8)
    scaled_test = te/variance.sqrt().clamp_min(1e-8)
    if train_targets is not None:
        train_targets = torch.as_tensor(train_targets, device=train.device, dtype=torch.float32)
        test_targets = torch.as_tensor(test_targets, device=train.device, dtype=torch.float32)
        global_pred = ridge_prediction(scaled_train[:, :rank], train_targets, scaled_test[:, :rank])
    rng = torch.Generator(device="cpu").manual_seed(2026090621)
    rows = []
    for i, query in enumerate(te):
        distances = (tr-query).square().sum(1)
        near = balanced_neighbors(distances, train_groups, neighbors, max_per_group)
        random_dist = torch.rand(len(tr), generator=rng)
        random = balanced_neighbors(random_dist, train_groups, neighbors, max_per_group)
        row = {"episode": str(test_groups[i]), "row": i,
               "global_reconstruction_fraction": float(global_error[i]/max(train_total, 1e-12)),
               "ambient_omitted_fraction": float(omitted[i]/max(train_total, 1e-12)),
               "nearest_train_distance": float(distances.min().sqrt()),
               "nearest_train_centered_cosine": float(torch.nn.functional.cosine_similarity(query[None], tr[near[:1]], dim=1)[0])}
        if train_targets is not None:
            row["global_target_squared_error"] = float((global_pred[i]-test_targets[i]).square().sum())
        for label, indices in (("local", near), ("random_neighborhood", random)):
            center, basis, eigenvalues = gram_pca(tr[indices], rank)
            if len(basis) != rank:
                raise ValueError("Local chart rank deficient; cannot silently compare unequal ranks")
            residual = query-center-(query-center)@basis.T@basis
            row[label+"_reconstruction_fraction"] = float((omitted[i]+residual.square().sum())/max(train_total, 1e-12))
            row[label+"_global_tangent_overlap"] = subspace_overlap(basis, global_basis)
            row[label+"_neighbors"] = indices
            row[label+"_neighbor_episodes"] = sorted(set(str(train_groups[j]) for j in indices))
            row[label+"_eigenvalues"] = eigenvalues.cpu().tolist()
            if train_targets is not None:
                # Identical fixed-rank representation and ridge penalty, using
                # only TRAIN neighbors. Locality is a piecewise nonlinear map.
                fit_x = (tr[indices]-center)@basis.T/eigenvalues.sqrt().clamp_min(1e-8)
                query_x = ((query-center)@basis.T/eigenvalues.sqrt().clamp_min(1e-8))[None]
                pred = ridge_prediction(fit_x, train_targets[indices], query_x)
                row[label+"_target_squared_error"] = float((pred[0]-test_targets[i]).square().sum())
        rows.append(row)
    groups = sorted(set(str(g) for g in test_groups))
    metrics = [k for k, v in rows[0].items() if isinstance(v, float)]
    by_episode = [{"episode": g, **{k: float(np.mean([r[k] for r in rows if r["episode"] == g])) for k in metrics}} for g in groups]
    return {"rows": rows, "episode_summary": by_episode, "independent_development_episodes": len(groups),
            "equal_episode_means": {k: float(np.mean([r[k] for r in by_episode])) for k in metrics},
            "chart_rank": rank, "ambient_rank": len(ambient), "neighbors": neighbors,
            "train_eigenvalues": variance.cpu().tolist(),
            "train_distribution": distribution_summary(scaled_train[:, :rank], train_groups),
            "development_distribution": distribution_summary(scaled_test[:, :rank], test_groups),
            "claims": "Exploratory local approximation and optional prediction; tangent rotation/reconstruction alone does not establish a curved causal manifold or steering utility",
            "metric": "Raw Euclidean after TRAIN centering/PCA; whitening used only for descriptive scores and fixed-rank ridge",
            "training_support": {"rows": len(train), "episodes": len(set(map(str, train_groups)))}}
