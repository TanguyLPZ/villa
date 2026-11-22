"""Plot the exact triangular mesh intersections on one public CT slice."""

import argparse
from base64 import b64encode
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import torch
from vc_reader import ScrollSource

from offset_inputs import SCAN
from run_offset_cpu import read_surface


def fitted_mesh(root, window, config):
    checkpoint = torch.load(root / f"{window}-{config}" / "model_final.pt",
                            weights_only=True, map_location="cpu")
    return checkpoint["mesh_flat"][:, 0].permute(1, 2, 0).numpy()


def intersect_grid(xyz, z, bounds):
    x0, x1, y0, y1 = bounds
    valid = np.isfinite(xyz).all(axis=-1) & (xyz > 0).all(axis=-1)
    nearby = (valid & (xyz[..., 0] >= x0 - 64) & (xyz[..., 0] < x1 + 64)
              & (xyz[..., 1] >= y0 - 64) & (xyz[..., 1] < y1 + 64)
              & (np.abs(xyz[..., 2] - z) < 64))
    rows, cols = np.where(nearby)
    if not len(rows):
        return []
    grid = xyz[max(0, rows.min() - 1):rows.max() + 2, max(0, cols.min() - 1):cols.max() + 2]
    a, b, c, d = grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:]
    segments = []
    for corners in ((a, b, d), (a, d, c)):
        tris = np.stack(corners, axis=-2).reshape(-1, 3, 3)
        selected = (np.isfinite(tris).all(axis=(1, 2)) & (tris > 0).all(axis=(1, 2))
                    & (tris[..., 2].min(axis=1) <= z) & (tris[..., 2].max(axis=1) > z))
        for tri in tris[selected]:
            endpoints = []
            for u, v in zip(tri, np.roll(tri, -1, axis=0)):
                if (u[2] <= z < v[2]) or (v[2] <= z < u[2]):
                    t = (z - u[2]) / (v[2] - u[2])
                    endpoints.append(u[:2] + t * (v[:2] - u[:2]))
            if len(endpoints) == 2:
                segments.append(endpoints)
    return segments


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work", required=True, type=Path)
    args = cli.parse_args()
    root = args.work
    plan = json.loads((root / "plan.json").read_text())
    field = json.loads((root / plan["manifest"]).read_text())
    window = "forward-1"
    start, _ = read_surface(root / f"{window}.tifxyz")
    target, target_valid = read_surface(root / "w031.tifxyz")
    target[~target_valid] = -1
    shipped = fitted_mesh(root, window, "shipped")
    candidate = fitted_mesh(root, window, "offset-dominant")
    surfaces = (start, shipped, candidate)
    points = np.concatenate([mesh.reshape(-1, 3) for mesh in surfaces])
    low = int(np.ceil(max(mesh[..., 2].min() for mesh in surfaces)))
    high = int(np.floor(min(mesh[..., 2].max() for mesh in surfaces)))
    if low > high:
        raise ValueError("No common axial slice intersects the three meshes")
    z = (low + high) // 2
    x0, y0 = np.floor(points.min(axis=0)[:2] - 56).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)[:2] + 56).astype(int)
    bounds = x0, x1, y0, y1
    volume = ScrollSource("PHerc0139", volume=SCAN).volume
    if volume is None or volume.id != SCAN:
        raise ValueError(f"Public CT scan {SCAN} was not resolved by the catalog")
    with volume.open(level=0) as reader:
        if tuple(reader.shape) != tuple(field["base_shape_zyx"]):
            raise ValueError(f"CT shape {reader.shape} differs from the fitting coordinate frame")
        ct = np.asarray(reader[z, y0:y1, x0:x1])
    black, white = np.percentile(ct, [3, 97])
    if black == white:
        raise ValueError(f"Flat CT intensity in sample at {bounds}, z={z}")
    start_lines = intersect_grid(start, z, bounds)
    target_lines = intersect_grid(target, z, bounds)
    if not start_lines or not target_lines:
        raise ValueError("The slice must show both traced reference windings")

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.25), layout="constrained")
    colors = {"w030": "#eeb14e", "w031": "#57dbe4", "fit": "#ff5f8b"}
    for axis, mesh, name in zip(axes, (shipped, candidate), ("Default Offset", "Retuned Offset")):
        fitted_lines = intersect_grid(mesh, z, bounds)
        if not fitted_lines:
            raise ValueError(f"No fit intersection for {name}")
        axis.imshow(ct, cmap="gray", vmin=black, vmax=white,
                    extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        for segments, color, style in ((start_lines, colors["w030"], "dashed"),
                                       (target_lines, colors["w031"], "dashed"),
                                       (fitted_lines, colors["fit"], "solid")):
            axis.add_collection(LineCollection(segments, colors=[color], linewidths=1.4, linestyles=style))
        axis.set(xlim=(x0, x1), ylim=(y1, y0), title=name)
        axis.set_axis_off()
        scale = 0.25 * 1000 / volume.pixel_size_um
        axis.plot([x0 + 18, x0 + 18 + scale], [y1 - 18, y1 - 18], color="white", linewidth=2)
        axis.text(x0 + 18, y1 - 27, "0.25 mm", color="white", fontsize=8)
    figure.suptitle(f"PHerc. 139 | CT axial slice z = {z} | Offset −1")
    handles = [Line2D([0], [0], color=colors[k], linestyle="-" if k == "fit" else "--", label=label)
               for k, label in (("w030", "Traced start (w030)"), ("w031", "Traced target (w031)"), ("fit", "Fitted mesh"))]
    axes[1].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    output = root / "PHerc0139_offset_cross_section.png"
    figure.savefig(output, dpi=120)
    plt.close(figure)
    print(f"OFFSET_CT scan={volume.id} z={z} xyz_bounds={bounds} pixel_size_um={volume.pixel_size_um}")
    encoded = b64encode(output.read_bytes()).decode("ascii")
    for i in range(0, len(encoded), 1800):
        print(f"OFFSET_PNG {i} {encoded[i:i + 1800]}")


if __name__ == "__main__":
    main()
