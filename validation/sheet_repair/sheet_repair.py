"""Repair a masked strip with two fixed boundaries and raw CT profiles."""

import numpy as np


OFFSETS = np.arange(-32, 33, 2, dtype=np.float32)
PROFILE = np.arange(-12, 13, 2, dtype=np.float32)


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1.e-6)


def normals(surface):
    along_rows, along_columns = np.gradient(surface, axis=(0, 1))
    return unit(np.cross(along_rows, along_columns))


def interpolate_strip(observed, start, stop):
    if not 2 <= start < stop <= observed.shape[1] - 2:
        raise ValueError("The missing strip needs two visible columns at each boundary")
    if not np.isnan(observed[:, start:stop]).all():
        raise ValueError("Mask the withheld strip before reconstruction")
    steps = stop - start + 1
    t = np.arange(1, steps, dtype=np.float32)[None, :, None] / steps
    left, right = observed[:, start - 1:start], observed[:, stop:stop + 1]
    slope_l = steps * (left - observed[:, start - 2:start - 1])
    slope_r = steps * (observed[:, stop + 1:stop + 2] - right)
    t2, t3 = t * t, t * t * t
    linear, cubic = observed.copy(), observed.copy()
    linear[:, start:stop] = left * (1 - t) + right * t
    cubic[:, start:stop] = ((2 * t3 - 3 * t2 + 1) * left + (t3 - 2 * t2 + t) * slope_l
                            + (-2 * t3 + 3 * t2) * right + (t3 - t2) * slope_r)
    if not np.isfinite(cubic).all():
        raise ValueError("Visible coordinates must be finite")
    return linear, cubic


def anchored_paths(costs, offsets):
    """Return paths and margins against a different path with the same endpoints."""
    rows, steps, states = costs.shape
    zero = int(np.flatnonzero(offsets == 0).item())
    difference = offsets[:, None] - offsets[None, :]
    transitions = np.where(np.abs(difference) <= 10, .035 * (difference / 2) ** 2, np.inf)
    forward = np.full((rows, steps, states), np.inf, dtype=np.float32)
    backward = forward.copy()
    predecessors = np.empty((rows, steps, states), dtype=np.int32)
    forward[:, 0, zero] = costs[:, 0, zero]
    for step in range(1, steps):
        choices = forward[:, step - 1, :, None] + transitions
        predecessors[:, step] = choices.argmin(axis=1)
        forward[:, step] = costs[:, step] + choices.min(axis=1)
    if not np.isfinite(forward[:, -1, zero]).all():
        raise ValueError("No complete path connects the visible boundaries")
    path = np.full((rows, steps), zero, dtype=np.int32)
    row = np.arange(rows)
    for step in range(steps - 1, 0, -1):
        path[:, step - 1] = predecessors[row, step, path[:, step]]

    backward[:, -1, zero] = costs[:, -1, zero]
    for step in range(steps - 2, -1, -1):
        choices = transitions + backward[:, step + 1, None, :]
        backward[:, step] = costs[:, step] + choices.min(axis=2)
    totals = np.subtract(forward + backward, costs, out=np.full_like(forward, np.inf),
                         where=np.isfinite(costs))
    separated = np.abs(offsets - offsets[path][..., None]) >= 10
    alternatives = np.min(np.where(separated, totals, np.inf), axis=-1)
    return path, alternatives - forward[:, -1, zero, None]


def profile_features(values):
    intensity = unit(values - values.mean(axis=-1, keepdims=True))
    derivative = unit(np.diff(values, axis=-1))
    return np.concatenate((.8 * intensity, .6 * derivative), axis=-1)


def fit_strip(observed, start, stop, sample):
    linear, cubic = interpolate_strip(observed, start, stop)
    directions = normals(cubic)
    visible = [start - 2, start - 1, stop, stop + 1]
    boundary_xyz = cubic[:, visible, None, :] + directions[:, visible, None, :] * PROFILE[None, None, :, None]
    boundary = sample(boundary_xyz)
    features = profile_features(boundary)
    left = unit(features[:, :2].mean(axis=1))
    right = unit(features[:, 2:].mean(axis=1))
    t = np.arange(1, stop - start + 1, dtype=np.float32)[None, :, None] / (stop - start + 1)
    template = unit(left[:, None] * (1 - t) + right[:, None] * t)
    agreement = np.clip((np.sum(left * right, axis=-1) + .25) / 1.25, 0, 1)
    contrast = np.clip(boundary.std(axis=-1).min(axis=-1) / 12, 0, 1)
    confidence = agreement * contrast

    center = cubic[:, start:stop]
    normal = directions[:, start:stop]
    distances = OFFSETS[:, None] + PROFILE[None, :]
    candidate_xyz = center[:, :, None, None, :] + normal[:, :, None, None, :] * distances[None, None, :, :, None]
    candidate = profile_features(sample(candidate_xyz))
    mismatch = 1 - np.sum(candidate * template[:, :, None], axis=-1)
    unary = (mismatch * confidence[:, None, None] + .08 * (OFFSETS / 16) ** 2).astype(np.float32)
    local_shift = OFFSETS[unary.argmin(axis=-1)]

    path, _ = anchored_paths(np.pad(unary, ((0, 0), (1, 1), (0, 0))), OFFSETS)
    shifts = OFFSETS[path[:, 1:-1]]
    padded = np.pad(shifts, ((1, 1), (0, 0)), mode="edge")
    desired = (padded[:-2] + padded[2:]) / 2
    joint = unary + .08 * ((OFFSETS - desired[..., None]) / 6) ** 2
    path, margin = anchored_paths(np.pad(joint, ((0, 0), (1, 1), (0, 0))), OFFSETS)
    connected_shift = OFFSETS[path[:, 1:-1]]
    uncertain = (confidence[:, None] < .25) | (margin[:, 1:-1] < .10)

    local, connected = cubic.copy(), cubic.copy()
    local[:, start:stop] = center + local_shift[..., None] * normal
    connected[:, start:stop] = center + connected_shift[..., None] * normal
    return {"linear": linear, "cubic": cubic, "ct_local": local, "ct_connected": connected}, uncertain
