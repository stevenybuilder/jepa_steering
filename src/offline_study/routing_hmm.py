"""Validated port of the archived fixed two-state imagined-history model.

Fitting accepts native model features only. Online routing accepts exactly H1/H2;
it never smooths over future observations or shares beliefs across candidates.
This module does not grant efficacy/eligibility or behavioral launch permission.
"""
import torch


def emission(x, mean, variance):
    return -.5 * (((x[..., None, :] - mean) ** 2 / variance) + variance.log()
                  + torch.log(torch.tensor(2 * torch.pi))).sum(-1)


def filter_beliefs(x, model):
    emit = emission(x, model["means"], model["variance"])
    state = model["initial"].log() + emit[:, 0]
    states = [state.softmax(-1)]
    for t in range(1, x.shape[1]):
        state = torch.logsumexp(state[:, :, None] + model["transition"].log()[None], 1) + emit[:, t]
        states.append(state.softmax(-1))
    return torch.stack(states, 1)


def validate_model(model):
    shapes = {"center": (400,), "basis": (3, 400), "scale": (3,), "means": (2, 3),
        "variance": (2, 3), "initial": (2,), "transition": (2, 2), "gates": (2,),
        "occupancy": (2,), "regime_drift": (2,)}
    for key, shape in shapes.items():
        value = model[key]
        if value.shape != shape or value.dtype != torch.float64 or not torch.isfinite(value).all():
            raise ValueError("Invalid fixed HMM parameter: " + key)
    if (not (model["scale"] > 0).all() or not (model["variance"] >= .05).all() or
            not (model["initial"] > 0).all() or not (model["transition"] > 0).all() or
            not (model["occupancy"] > 0).all() or
            sorted(model["gates"].tolist()) != [.5, 1.] or
            len(model["log_likelihood_history"]) != 20 or model["training_length"] != 6):
        raise ValueError("Changed HMM floors, gates, horizon or iteration count")
    for value in (model["initial"].sum(), model["occupancy"].sum(), model["transition"].sum(-1)):
        if not torch.allclose(value, torch.ones_like(value), atol=1e-10, rtol=1e-10):
            raise ValueError("Invalid HMM probability normalization")


@torch.no_grad()
def fit_hmm(features):
    """Exactly PCA3, two diagonal states,20EM steps, Dirichlet1/floor.05."""
    if (features.ndim != 3 or features.shape[1:] != (6, 400) or len(features) < 2 or
            features.device.type != "cpu" or not torch.isfinite(features).all()):
        raise ValueError("Require finite CPU native H1-H6 histories with400features")
    n, t, d = features.shape
    flat = features.reshape(-1, d).double()
    center = flat.mean(0)
    _, _, components = torch.linalg.svd(flat - center, full_matrices=False)
    basis = components[:3]
    scale = ((flat - center) @ basis.T).std(0, unbiased=False).clamp_min(1e-8)
    x = ((features.double() - center) @ basis.T) / scale
    order = torch.argsort(x.reshape(-1, 3)[:, 0])
    half = len(order) // 2
    means = torch.stack([x.reshape(-1, 3)[order[:half]].mean(0), x.reshape(-1, 3)[order[half:]].mean(0)])
    variance = torch.ones(2, 3, dtype=torch.float64)
    initial = torch.ones(2, dtype=torch.float64) / 2
    transition = torch.tensor([[.9, .1], [.1, .9]], dtype=torch.float64)
    history = []
    for _ in range(20):
        emit = emission(x, means, variance)
        forward = [initial.log() + emit[:, 0]]
        for j in range(1, t):
            forward.append(torch.logsumexp(forward[-1][:, :, None] + transition.log()[None], 1) + emit[:, j])
        alpha = torch.stack(forward, 1)
        normalizer = torch.logsumexp(alpha[:, -1], -1)
        beta = torch.zeros_like(alpha)
        for j in range(t - 2, -1, -1):
            beta[:, j] = torch.logsumexp(transition.log()[None] + emit[:, j + 1, None, :] + beta[:, j + 1, None, :], -1)
        gamma = (alpha + beta - normalizer[:, None, None]).exp()
        xi = (alpha[:, :-1, :, None] + transition.log()[None, None] + emit[:, 1:, None, :]
              + beta[:, 1:, None, :] - normalizer[:, None, None, None]).exp().sum((0, 1))
        initial = (gamma[:, 0].sum(0) + 1) / (n + 2)
        transition = (xi + 1) / (xi.sum(1, keepdim=True) + 2)
        count = gamma.sum((0, 1))
        means = (gamma[..., None] * x[:, :, None, :]).sum((0, 1)) / count[:, None]
        variance = (gamma[..., None] * (x[:, :, None, :] - means).square()).sum((0, 1)) / count[:, None]
        variance = variance.clamp_min(.05)
        history.append(float(normalizer.sum()))
    model = dict(center=center, basis=basis, scale=scale, means=means,
                 variance=variance, initial=initial, transition=transition)
    beliefs = filter_beliefs(x, model)
    drift = torch.cat([torch.zeros(n, 1), (x[:, 1:] - x[:, :-1]).norm(dim=-1)], 1)
    regime_drift = (beliefs * drift[:, :, None]).sum((0, 1)) / beliefs.sum((0, 1))
    gates = torch.ones(2, dtype=torch.float64)
    gates[regime_drift.argmax()] = .5
    model.update(gates=gates, occupancy=beliefs.mean((0, 1)), regime_drift=regime_drift,
        log_likelihood_history=history, training_sequences=n, training_length=t)
    validate_model(model)
    return model


@torch.no_grad()
def route_prefix(prefix, model):
    """A stateless fresh filter for each candidate; future data are rejected."""
    validate_model(model)
    if (prefix.ndim != 3 or prefix.shape[1:] != (2, 400) or len(prefix) < 1 or
            not torch.isfinite(prefix).all()):
        raise ValueError("Online H3 routing accepts exactly native H1/H2, not future horizons")
    if prefix.device != model["center"].device:
        raise ValueError("Stage the model once on the prefix device")
    x = ((prefix.double() - model["center"]) @ model["basis"].T) / model["scale"]
    posterior = filter_beliefs(x, model)[:, -1]
    memoryless = (emission(x[:, 1], model["means"], model["variance"]) + model["occupancy"].log()).softmax(-1)
    static = model["occupancy"] @ model["gates"]
    return {"static": static.expand(len(prefix)), "hmm": posterior @ model["gates"],
        "memoryless": memoryless @ model["gates"], "posterior": posterior}
