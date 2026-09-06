"""Fixed-sample action-path geometry; no fitted manifold or steering claim.

The four interpolation inputs exclude the central activation by construction.
Curvature diagnostics use all five natural samples, but never fit an edit.
"""
from __future__ import annotations

import math
import torch


def central_estimates(noncentral: torch.Tensor) -> dict[str, torch.Tensor]:
    """Estimate t=0 using only samples at [-1, -.5, .5, 1].

    The cubic and least-squares affine estimators see exactly the same four samples.
    At this symmetric midpoint, the cubic estimate also equals quadratic LS.
    Cubic interpolation has negative weights and can overshoot their convex hull.
    Its endpoint-line projection isolates nonlinear speed from off-line geometry.
"""
    if noncentral.ndim < 2 or len(noncentral) != 4:
        raise ValueError("Exactly four noncentral samples required")
    if not noncentral.is_floating_point() or not torch.isfinite(noncentral).all():
        raise ValueError("Finite floating-point activations required")
    x = noncentral.double()
    origin = x[0]
    centered = x-origin
    # Centered arithmetic preserves constant fields despite large common offsets.
    estimates = {
        "endpoint_chord": origin+.5*centered[3],
        "near_chord": origin+.5*(centered[1]+centered[2]),
        "linear_equal_data": origin+centered.mean(0),
        "cubic_equal_data": origin+(2/3)*(centered[1]+centered[2])-(1/6)*centered[3],
    }
    chord = centered[3]
    norm_squared = chord.square().sum()
    if float(norm_squared) > 1e-24:
        coefficient = ((estimates["cubic_equal_data"]-origin)*chord).sum()/norm_squared
        line = origin+coefficient*chord
    else:
        # Closed/degenerate endpoint lines are separately flagged by geometry;
        # this is a point-reference control, not a well-defined chord direction.
        line = origin
    estimates["reparameterized_chord"] = line
    estimates["reflected_curvature"] = 2*line-estimates["cubic_equal_data"]
    return {name: value.to(noncentral.dtype) for name, value in estimates.items()}


def interior_estimates(noncentral: torch.Tensor, t: float) -> dict[str, torch.Tensor]:
    """Frozen four-donor interpolation at an unseen interior coordinate.

No target activation is accepted. The near chord only supports [-.5,.5];
all other methods use the same original nonzero samples as central_estimates.
"""
    if not math.isfinite(t) or not -.5 <= t <= .5:
        raise ValueError("Evaluation must remain inside the two nearest donors")
    if t == 0:
        return central_estimates(noncentral)
    if noncentral.ndim < 2 or len(noncentral) != 4:
        raise ValueError("Exactly four noncentral samples required")
    if not noncentral.is_floating_point() or not torch.isfinite(noncentral).all():
        raise ValueError("Finite floating-point activations required")
    x = noncentral.double()
    origin = x[0]
    centered = x-origin
    coordinates = (-1., -.5, .5, 1.)
    weights = [math.prod((t-other)/(node-other) for other in coordinates if other != node)
               for node in coordinates]
    shape = (4,)+(1,)*(x.ndim-1)
    coefficient = torch.tensor(weights, device=x.device, dtype=x.dtype).reshape(shape)
    nodes = torch.tensor(coordinates, device=x.device, dtype=x.dtype).reshape(shape)
    estimates = {
        "endpoint_chord": origin+((t+1)/2)*centered[3],
        "near_chord": x[1]+(t+.5)*(x[2]-x[1]),
        "linear_equal_data": origin+centered.mean(0)+(t/2.5)*(nodes*centered).sum(0),
        "cubic_equal_data": origin+(coefficient*centered).sum(0),
    }
    chord = centered[3]
    norm_squared = chord.square().sum()
    if float(norm_squared) > 1e-24:
        position = ((estimates["cubic_equal_data"]-origin)*chord).sum()/norm_squared
        line = origin+position*chord
    else:
        line = origin
    estimates["reparameterized_chord"] = line
    estimates["reflected_curvature"] = 2*line-estimates["cubic_equal_data"]
    return {name: value.to(noncentral.dtype) for name, value in estimates.items()}


def sampled_path_geometry(samples: torch.Tensor) -> dict:
    """Describe five ordered samples at [-1,-.5,0,.5,1] in raw coordinates.

Chord-orthogonal deviation distinguishes bending from varying speed along a
straight line. Tangent reversals and arc/chord alone can also reflect backtracking.
These finite-sample diagnostics do not estimate a full representation manifold.
"""
    if samples.ndim < 2 or len(samples) != 5:
        raise ValueError("Exactly five ordered samples required")
    if not samples.is_floating_point() or not torch.isfinite(samples).all():
        raise ValueError("Finite floating-point activations required")
    x = samples.double().reshape(5, -1)
    x = x-x[0]
    segments = x[1:]-x[:-1]
    lengths = segments.norm(dim=1)
    path_length = float(lengths.sum())
    chord = x[-1]
    chord_length = float(chord.norm())
    threshold = max(path_length*1e-12, 1e-15)
    chord_valid = chord_length > threshold
    if chord_valid:
        unit = chord/chord_length
        coordinate = x@unit
        orthogonal = x-coordinate[:, None]*unit
        orthogonal_norm = orthogonal.norm(dim=1)
        fraction = (coordinate/chord_length).tolist()
        backward = int(((coordinate[1:]-coordinate[:-1]) < -threshold).sum())
    else:
        orthogonal_norm = None
        fraction = None
        backward = None
    angles = []
    normal_acceleration = []
    curvature = []
    for i in range(3):
        if float(lengths[i]) > threshold and float(lengths[i+1]) > threshold:
            cosine = (segments[i]@segments[i+1])/(lengths[i]*lengths[i+1])
            angles.append(float(torch.acos(cosine.clamp(-1, 1))))
        else:
            angles.append(None)
        # dt=.5: central velocity denominator=1; acceleration denominator=.25.
        velocity = x[i+2]-x[i]
        acceleration = 4*(x[i+2]-2*x[i+1]+x[i])
        speed = float(velocity.norm())
        if speed > threshold:
            tangent = velocity/speed
            normal = acceleration-(acceleration@tangent)*tangent
            normal_acceleration.append(float(normal.norm()))
            curvature.append(float(normal.norm())/(speed*speed))
        else:
            normal_acceleration.append(None)
            curvature.append(None)
    return {
        "sample_t": [-1, -.5, 0, .5, 1],
        "path_length": path_length,
        "chord_length": chord_length,
        "path_chord_ratio": path_length/chord_length if chord_valid else None,
        "chord_orthogonal_distance": orthogonal_norm.tolist() if chord_valid else None,
        "max_orthogonal_fraction_of_chord": float(orthogonal_norm.max())/chord_length if chord_valid else None,
        "chord_coordinate_fraction": fraction,
        "backtracking_segments": backward,
        "successive_segment_angle_radians": angles,
        "normal_second_difference": normal_acceleration,
        "finite_sample_path_curvature": curvature,
        "degenerate_endpoint_chord": not chord_valid,
        "metric": "Raw full-spatial Euclidean; no pooling, PCA, whitening, or fitted readout",
        "limitations": "A sampled action-response path, not proof of a manifold, semantic coordinate, or useful control",
    }
