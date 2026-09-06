"""Fixed TRAIN-only residual estimators; no physical test labels enter this API.

NumPy and Torch inputs are accepted. Results are float32 Torch tensors on the
input tensor device (CPU if all inputs are NumPy). Gram calculations use float64.
"""
from __future__ import annotations

import math
import torch


def _device(*values):
    devices = {x.device for x in values if isinstance(x, torch.Tensor)}
    if len(devices) > 1:
        raise ValueError("All tensor inputs must use one device")
    return next(iter(devices), torch.device("cpu"))


def _tensor(value, device, dtype=torch.float64):
    result = torch.as_tensor(value, device=device, dtype=dtype).detach()
    if not torch.isfinite(result).all():
        raise ValueError("Inputs must be finite")
    return result


def _groups(values, count):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = list(values)
    if len(values) != count or any(isinstance(x, (list, tuple, dict)) for x in values):
        raise ValueError("train_group must have one scalar group per TRAIN row")
    mapping = {}
    encoded = []
    for value in values:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Groups must be finite")
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return encoded, len(mapping)


def _distance2(x, y):
    return torch.cdist(x, y, p=2).square()


def _positive_median(values):
    positive = values[values > 1e-12]
    return positive.median() if positive.numel() else values.new_tensor(1.0)


@torch.no_grad()
def fit_predict(train_x, train_residual, test_x, train_group, family: str):
    """Fit a fixed residual estimator exclusively from supplied TRAIN rows.

    Families: linear/linear_ridge, rbf/rbf_ridge, knn/state_diverse_knn.
    RBF uses exp(-squared_distance / median_positive_TRAIN_squared_distance).
    kNN admits at most two samples per group and at most eight overall; if fewer
    than eight are available it retains the smaller diverse set, never relaxing
    the group cap. Its normalized weights are exp(-distance/TRAIN_median_distance).
    """
    aliases = {"linear_ridge": "linear", "rbf_ridge": "rbf",
               "state_diverse_knn": "knn", "weighted_knn": "knn"}
    family = aliases.get(family, family)
    if family not in ("linear", "rbf", "knn"):
        raise ValueError("Unknown fixed family")
    device = _device(train_x, train_residual, test_x)
    x = _tensor(train_x, device)
    y = _tensor(train_residual, device)
    test = _tensor(test_x, device)
    if x.ndim != 2 or y.ndim != 2 or test.ndim != 2:
        raise ValueError("Features and residuals must be matrices")
    if min(x.shape) < 1 or y.shape[0] != x.shape[0] or y.shape[1] < 1 or test.shape[1] != x.shape[1]:
        raise ValueError("TRAIN/TEST dimensions differ or TRAIN is empty")
    groups, group_count = _groups(train_group, len(x))
    mean = x.mean(0)
    scale = x.std(0, unbiased=False)
    floored = scale < 1e-5
    scale = torch.where(floored, torch.ones_like(scale), scale)
    x = (x - mean) / scale
    test = (test - mean) / scale
    ymean = y.mean(0)
    centered_y = y - ymean
    metadata = dict(family=family, train_rows=len(x), test_rows=len(test),
                    train_groups=group_count, features=x.shape[1], outputs=y.shape[1],
                    standardization="TRAIN population mean/std; std<1e-5 replaced by1",
                    floored_features=int(floored.sum()),
                    train_feature_mean=mean.cpu().tolist(),
                    train_feature_scale=scale.cpu().tolist(),
                    residual_intercept="TRAIN mean", gram_dtype="float64")
    if family in ("linear", "rbf"):
        if family == "linear":
            kernel = (x @ x.T) / x.shape[1]
            cross = (test @ x.T) / x.shape[1]
        else:
            distance = _distance2(x, x)
            bandwidth = _positive_median(distance)
            kernel = torch.exp(-distance / bandwidth)
            cross = torch.exp(-_distance2(test, x) / bandwidth)
            metadata["bandwidth_squared"] = float(bandwidth)
        regularization = 0.1 * torch.diagonal(kernel).mean()
        metadata["ridge_regularization"] = float(regularization)
        if float(regularization) == 0.0:
            # All standardized linear features are zero: only the intercept is
            # identifiable. Do not invent a different ridge penalty.
            prediction = ymean.expand(len(test), -1).clone()
            metadata["degenerate_intercept_only"] = True
        else:
            system = kernel + regularization * torch.eye(len(x), device=device, dtype=torch.float64)
            # Solve against M test vectors, avoiding an N-by-O coefficient solve
            # when visual residual output dimension is large.
            weights = torch.linalg.solve(system, cross.T).T
            prediction = weights @ centered_y + ymean
            metadata["degenerate_intercept_only"] = False
    else:
        train_distances = _distance2(x, x).sqrt()
        bandwidth = _positive_median(train_distances)
        distance = _distance2(test, x).sqrt()
        rows, selections = [], []
        for row in distance:
            ordered = torch.argsort(row, stable=True).cpu().tolist()
            selected, counts = [], {}
            for index in ordered:
                group = groups[index]
                if counts.get(group, 0) >= 2:
                    continue
                selected.append(index)
                counts[group] = counts.get(group, 0) + 1
                if len(selected) == 8:
                    break
            index = torch.tensor(selected, device=device, dtype=torch.long)
            local_distance = row[index]
            weights = torch.exp(-(local_distance - local_distance.min()) / bandwidth)
            weights /= weights.sum()
            rows.append(weights @ centered_y[index] + ymean)
            selections.append(selected)
        prediction = torch.stack(rows) if rows else y.new_empty((0, y.shape[1]))
        metadata.update(bandwidth_distance=float(bandwidth), neighbors_max=8,
                        neighbors_per_group_max=2, selected_train_indices=selections,
                        selected_counts=[len(row) for row in selections])
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite fitted prediction")
    return prediction.float(), metadata


def _rows(value, device):
    value = _tensor(value, device)
    if value.ndim < 2:
        raise ValueError("Expected a leading row axis and at least one feature axis")
    return value, value.flatten(1)


@torch.no_grad()
def cap_edit(delta, predicted_visual, fraction):
    """Cap each raw edit at fraction * that row's predicted visual L2 norm."""
    fraction = float(fraction)
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("fraction must be finite and nonnegative")
    device = _device(delta, predicted_visual)
    original, change = _rows(delta, device)
    _, visual = _rows(predicted_visual, device)
    if change.shape != visual.shape:
        raise ValueError("Visual/edit row dimensions differ")
    radius = fraction * torch.linalg.vector_norm(visual, dim=1)
    length = torch.linalg.vector_norm(change, dim=1)
    factor = torch.minimum(torch.ones_like(length), radius / length.clamp_min(1e-30))
    return (change * factor[:, None]).reshape(original.shape).float()


@torch.no_grad()
def norm_match(sham, semantic):
    """Match row L2 norms; a zero sham cannot represent a nonzero target norm."""
    device = _device(sham, semantic)
    original, sham_rows = _rows(sham, device)
    _, semantic_rows = _rows(semantic, device)
    if sham_rows.shape != semantic_rows.shape:
        raise ValueError("Sham/semantic dimensions differ")
    source = torch.linalg.vector_norm(sham_rows, dim=1)
    target = torch.linalg.vector_norm(semantic_rows, dim=1)
    if torch.any((source == 0) & (target > 0)):
        raise ValueError("Cannot norm-match a zero sham to a nonzero semantic edit")
    return (sham_rows * (target / source.clamp_min(1e-30))[:, None]).reshape(original.shape).float()


@torch.no_grad()
def corrected_cost(native_cost, pred_visual, goal_visual, delta):
    """Replace only native mean-squared visual error, leaving proprio unchanged.

    native + [2*(pred-goal) dot delta + ||delta||^2]/O preserves the supplied
    native scalar exactly for zero edit, avoiding baseline reconstruction drift.
    Goal may be one flattened vector or one/batched visual row. Cost retains the
    native floating dtype so zero edits preserve supplied scalars exactly.
    """
    device = _device(native_cost, pred_visual, goal_visual, delta)
    _, pred = _rows(pred_visual, device)
    _, edit = _rows(delta, device)
    if pred.shape != edit.shape:
        raise ValueError("Prediction/edit dimensions differ")
    goal = _tensor(goal_visual, device)
    if goal.numel() == pred.shape[1]:
        goal = goal.reshape(1, -1).expand_as(pred)
    elif goal.ndim >= 2 and goal.shape[0] == pred.shape[0] and goal.numel() == pred.numel():
        goal = goal.reshape_as(pred)
    else:
        raise ValueError("Goal must broadcast by row without changing visual support")
    native_input = torch.as_tensor(native_cost, device=device)
    native_dtype = native_input.dtype if native_input.is_floating_point() else torch.float32
    native = _tensor(native_input, device, dtype=native_dtype)
    if native.ndim != 1 or native.shape[0] != pred.shape[0]:
        raise ValueError("native_cost must contain one scalar per row")
    correction = (2 * (pred - goal) * edit + edit.square()).mean(1)
    return native + correction.to(native_dtype)
