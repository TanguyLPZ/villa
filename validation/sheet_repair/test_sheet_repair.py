"""Check anchor preservation and continuity with a reproducible CT model."""

import unittest

import numpy as np

from sheet_repair import OFFSETS, anchored_paths, fit_strip, interpolate_strip


class SheetRepairTest(unittest.TestCase):
    def setUp(self):
        rows, cols = np.mgrid[:5, :18].astype(np.float32)
        self.surface = np.stack((100 + 4 * cols, 100 + 4 * rows, np.full_like(rows, 100)), axis=-1)
        self.start, self.stop = 3, 15
        self.observed = self.surface.copy()
        self.observed[:, self.start:self.stop] = np.nan

    def test_withheld_coordinates_must_be_masked(self):
        with self.assertRaisesRegex(ValueError, "Mask the withheld strip"):
            interpolate_strip(self.surface, self.start, self.stop)

    def test_connected_path_has_fixed_anchors_and_bounded_changes(self):
        cost = np.broadcast_to((OFFSETS - 18) ** 2, (2, 7, len(OFFSETS))).copy()
        path, margin = anchored_paths(cost, OFFSETS)
        route = OFFSETS[path]
        np.testing.assert_array_equal(route[:, [0, -1]], np.zeros((2, 2)))
        self.assertLessEqual(np.abs(np.diff(route, axis=1)).max(), 10)
        self.assertGreaterEqual(margin.min(), 0)

    def test_fit_tracks_off_plane_signal_with_visible_boundary_unchanged(self):
        def ridge(xyz):
            x, z = xyz[..., 0], xyz[..., 2]
            height = 100 + 8 * np.sin(np.pi * np.clip((x - 108) / 52, 0, 1)) ** 2
            d = z - height
            return (35 + 80 * np.exp(-.5 * (d / 2.5) ** 2)).astype(np.float32)

        result, review = fit_strip(self.observed, self.start, self.stop, ridge)
        connected = result["ct_connected"]
        np.testing.assert_array_equal(connected[:, :self.start], self.surface[:, :self.start])
        np.testing.assert_array_equal(connected[:, self.stop:], self.surface[:, self.stop:])
        self.assertGreaterEqual(float(connected[:, 8, 2].min()), 106)
        self.assertEqual(review.shape, (5, 12))


if __name__ == "__main__":
    unittest.main()
