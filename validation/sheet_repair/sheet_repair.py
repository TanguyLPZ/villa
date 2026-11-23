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


def anchored_paths(costs, offsets, *, pairs=None, max_step=10, smoothness=.035):
    """Return paths and margins against a different path with the same endpoints."""
    rows, steps, states = costs.shape
    zero = int(np.flatnonzero(offsets == 0).item())
    difference = offsets[:, None] - offsets[None, :]
    transitions = np.where(np.abs(difference) <= max_step, smoothness * (difference / 2) ** 2, np.inf)
    forward = np.full((rows, steps, states), np.inf, dtype=np.float32)
    backward = forward.copy()
    predecessors = np.empty((rows, steps, states), dtype=np.int32)
    forward[:, 0, zero] = costs[:, 0, zero]
    for step in range(1, steps):
        choices = forward[:, step - 1, :, None] + transitions
        if pairs is not None:
            choices += pairs[:, step - 1]
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
        if pairs is not None:
            choices += pairs[:, step]
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


def cubic_from_four_columns(observed, start, stop):
    """Estimate each end tangent from four measured intervals, with a fixed endpoint."""
    left, right = observed[:, start - 1:start], observed[:, stop:stop + 1]
    distances = np.arange(1, 5, dtype=np.float32)
    design = np.stack((distances, distances ** 2), axis=1)
    weights = np.linalg.pinv(design)[0]
    leading = np.einsum("c,rci->ri", weights, left - observed[:, start-5:start-1][:, ::-1])
    trailing = np.einsum("c,rci->ri", weights, observed[:, stop+1:stop+5] - right)
    steps = stop - start + 1
    t = np.arange(1, steps, dtype=np.float32)[None, :, None] / steps
    t2, t3 = t * t, t * t * t
    fitted = observed.copy()
    fitted[:, start:stop] = ((2 * t3 - 3 * t2 + 1) * left + (t3 - 2 * t2 + t) * steps * leading[:, None]
                              + (-2 * t3 + 3 * t2) * right + (t3 - t2) * steps * trailing[:, None])
    return fitted


def trace_continuous(cubic, start, stop, sample):
    """Track the CT profile between adjacent samples at finer than mesh spacing."""
    subdivisions = 3
    spacing = stop - start + 1
    t = np.arange(spacing * subdivisions + 1, dtype=np.float32) / subdivisions
    lower = np.minimum(t.astype(np.int32), spacing - 1)
    fraction = (t - lower)[None, :, None].astype(np.float32)
    columns = start - 1 + lower
    normal = normals(cubic)
    center = cubic[:, columns] * (1 - fraction) + cubic[:, columns + 1] * fraction
    direction = unit(normal[:, columns] * (1 - fraction) + normal[:, columns + 1] * fraction)
    profile = np.arange(-8, 9, 2, dtype=np.float32)
    distances = OFFSETS[:, None] + profile[None, :]
    xyz = center[:, :, None, None] + direction[:, :, None, None] * distances[None, None, :, :, None]
    values = sample(xyz)
    features = profile_features(values)
    strength = np.clip(values.std(axis=-1) / 12, 0, 1).astype(np.float32)
    weighted = features * np.sqrt(strength)[..., None]
    similarity = np.matmul(weighted[:, :-1], weighted[:, 1:].swapaxes(-1, -2))
    pairs = -.45 * similarity
    zero = len(OFFSETS) // 2
    left = np.sum(weighted * weighted[:, :1, zero:zero + 1], axis=-1)
    right = np.sum(weighted * weighted[:, -1:, zero:zero + 1], axis=-1)
    blend = (t / spacing)[None, :, None]
    unary = (.035 * (OFFSETS / 14) ** 2 - .08 * ((1 - blend) * left + blend * right)).astype(np.float32)
    path, _ = anchored_paths(unary, OFFSETS, pairs=pairs, max_step=4, smoothness=.08)
    shifts = OFFSETS[path]
    padded = np.pad(shifts, ((1, 1), (0, 0)), mode="edge")
    desired = (padded[:-2] + padded[2:]) / 2
    joint = unary + .06 * ((OFFSETS - desired[..., None]) / 6) ** 2
    path, margin = anchored_paths(joint, OFFSETS, pairs=pairs, max_step=4, smoothness=.08)
    at_vertex = slice(subdivisions, -1, subdivisions)
    corrected = cubic.copy()
    corrected[:, start:stop] = center[:, at_vertex] + OFFSETS[path[:, at_vertex], None] * direction[:, at_vertex]
    first, last = strength[:, 0, zero], strength[:, -1, zero]
    uncertain = (np.minimum(first, last)[:, None] < .25) | (margin[:, at_vertex] < .15)
    return corrected, uncertain
