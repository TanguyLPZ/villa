"""Transfer local relief from a known adjacent winding, with fixed strip boundaries."""

import numpy as np
from scipy.spatial import cKDTree

from sheet_repair import normals


CELL_OFFSETS = np.array([[-1, -1], [-1, 0], [0, -1], [0, 0]])
TRIANGLES = np.array([[[0, 0], [0, 1], [1, 1]], [[0, 0], [1, 1], [1, 0]]])


class Winding:
    def __init__(self, surface, valid):
        self.surface = surface
        self.valid = valid
        self.indices = np.column_stack(np.nonzero(valid)).astype(np.int32)
        self.tree = cKDTree(surface[valid])

    def crossings(self, center, direction, expected):
        """Intersect each normal line with nearby triangles of this winding."""
        shape = expected.shape
        points = center.reshape(-1, 3).astype(np.float64)
        rays = direction.reshape(-1, 3).astype(np.float64)
        hint = expected.reshape(-1, 1)
        _, closest = self.tree.query(points + hint * rays, k=6)
        cells = self.indices[closest, None, :] + CELL_OFFSETS[None, None, :, :]
        coords = cells[..., None, None, :] + TRIANGLES[None, None, None, :, :, :]
        coords = coords.reshape(len(points), -1, 3, 2)
        outside = (coords < 0) | (coords >= np.asarray(self.valid.shape))
        rows, cols = np.moveaxis(np.clip(coords, 0, np.asarray(self.valid.shape) - 1), -1, 0)
        valid = ~outside.any(axis=(-2, -1)) & self.valid[rows, cols].all(axis=-1)
        a, b, c = np.moveaxis(self.surface[rows, cols].astype(np.float64), 2, 0)
        edge1, edge2 = b - a, c - a
        h = np.cross(rays[:, None], edge2)
        determinant = np.sum(edge1 * h, axis=-1)
        area = np.linalg.norm(np.cross(edge1, edge2), axis=-1)
        valid &= (area > 1e-6) & (np.abs(determinant) > .6 * area)
        reciprocal = np.divide(1, determinant, out=np.zeros_like(determinant), where=valid)
        s = points[:, None] - a
        u = np.sum(s * h, axis=-1) * reciprocal
        q = np.cross(s, edge1)
        v = np.sum(rays[:, None] * q, axis=-1) * reciprocal
        t = np.sum(edge2 * q, axis=-1) * reciprocal
        error = np.abs(t - hint)
        valid &= (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (error <= 36)
        available = valid.any(axis=1)
        index = np.argmin(np.where(valid, error, np.inf), axis=1)
        signed = t[np.arange(len(t)), index]
        return np.where(available, signed, np.nan).reshape(shape).astype(np.float32)

    def relief(self, center, directions, start, stop):
        window = slice(start - 2, stop + 2)
        bridge, rays = center[:, window], directions[:, window]
        at_ends = np.array([1, bridge.shape[1] - 2])
        anchors = bridge[:, at_ends]
        _, indices = self.tree.query(anchors.reshape(-1, 3))
        near = self.tree.data[indices].reshape(anchors.shape)
        initial = np.sum((near - anchors) * rays[:, at_ends], axis=-1)
        phases = np.linspace(-1 / (stop - start + 1), 1 + 1 / (stop - start + 1), bridge.shape[1],
                             dtype=np.float32)[None, :]
        expected = initial[:, :1] * (1 - phases) + initial[:, 1:] * phases
        offset = self.crossings(bridge, rays, expected)
        first, last = offset[:, 1:2], offset[:, -2:-1]
        spacing = first * (1 - phases[:, 2:-2]) + last * phases[:, 2:-2]
        face = np.sign(first)
        safe = (np.abs(spacing) >= 12) & (np.abs(spacing) <= 145) & (np.sign(last) == face)
        safe &= (np.abs(offset[:, 1:2] - offset[:, :1]) <= 8) & (np.abs(last - offset[:, -1:]) <= 8)
        measured = offset[:, 2:-2]
        safe &= np.sign(measured) == face
        return np.where(safe, measured - spacing, np.nan)


def guide_strip(cubic, start, stop, windings):
    """Use consistent donor relief where reliable crossings exist."""
    directions = normals(cubic)
    candidates = np.stack([winding.relief(cubic, directions, start, stop) for winding in windings])
    count = np.isfinite(candidates).sum(axis=0)
    correction = np.nansum(candidates, axis=0) / np.maximum(count, 1)
    if len(windings) == 2:
        correction[np.abs(candidates[0] - candidates[1]) > 10] = 0
    t = np.arange(1, stop - start + 1, dtype=np.float32) / (stop - start + 1)
    shifts = .6 * np.clip(correction, -24, 24) * np.sin(np.pi * t) ** 2
    result = cubic.copy()
    result[:, start:stop] += shifts[..., None] * directions[:, start:stop]
    return result, np.abs(shifts) >= 1
