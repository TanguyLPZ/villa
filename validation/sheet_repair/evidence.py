"""Save the preselected example as TIFXYZ surfaces and a CT cross-section."""

import json
from uuid import NAMESPACE_URL, uuid5

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from public_data import BUCKET, MESHES, VOXEL_UM
from sheet_repair import normals, unit


def write_surface(out, name, surface, source, case):
    path = out / f"{name}.tifxyz"
    path.mkdir(exist_ok=True)
    valid = np.isfinite(surface).all(axis=-1)
    coords = np.where(valid[..., None], surface, -1)
    for i, axis in enumerate("xyz"):
        tifffile.imwrite(path / f"{axis}.tif", coords[..., i].astype(np.float32))
    metadata = {
        "uuid": str(uuid5(NAMESPACE_URL, f"{BUCKET}/{MESHES[case['mesh']]}{case['row']},{case['col']}/{name}")),
        "type": "seg",
        "format": "tifxyz",
        "name": name,
        "scale": source.get("scale", [1, 1]),
        "bbox": [coords[valid].min(axis=0).tolist(), coords[valid].max(axis=0).tolist()],
    }
    (path / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_evidence(out, sample, case, source, start, stop, truth, repaired, uncertain):
    for name, surface in {"reference": truth, **repaired}.items():
        write_surface(out, name, surface, source, case)
    observed = truth.copy()
    observed[:, start:stop] = np.nan
    write_surface(out, "masked_input", observed, source, case)
    (out / "example.json").write_text(json.dumps({**case, "hidden_columns": [start, stop],
                                                  "review_mask": uncertain.tolist()}, indent=2) + "\n",
                                    encoding="utf-8")

    cubic = repaired["cubic"]
    middle = truth.shape[0] // 2
    rail = cubic[middle]
    direction = normals(cubic)[middle]
    columns = np.arange(len(rail))
    subcolumns = np.linspace(0, columns[-1], 4 * len(rail) - 3)
    offsets = np.arange(-50, 50.1, .5, dtype=np.float32)
    centers = np.stack([np.interp(subcolumns, columns, axis) for axis in rail.T], axis=-1)
    vectors = unit(np.stack([np.interp(subcolumns, columns, axis) for axis in direction.T], axis=-1))
    section = sample(centers[None, :, :] + offsets[:, None, None] * vectors[None, :, :])
    if np.percentile(section, 99) == 0:
        raise ValueError("No usable raw CT signal in the fixed evidence example")
    cut = slice(start, stop)
    reference_delta = truth[middle, cut] - rail[cut]
    target_curve = np.sum(reference_delta * direction[cut], axis=-1)
    off_plane = np.linalg.norm(reference_delta - target_curve[:, None] * direction[cut], axis=-1)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, sharey=True, layout="constrained")
    fig.suptitle(f"PHerc. 139 • {case['id']} • {stop - start}-vertex gap\n"
                 f"Curved CT section at mesh row {case['row'] + middle} • {VOXEL_UM} µm/voxel")
    levels = np.percentile(section, [2, 98])
    extent = [-.125, columns[-1] + .125, offsets[0] - .25, offsets[-1] + .25]
    for ax, method, title in zip(axs, ("cubic", "ct_continuous"),
                                 ("Cubic interpolation", "CT tracing between adjacent profiles"), strict=True):
        ax.imshow(section, cmap="gray", vmin=levels[0], vmax=levels[1], origin="lower",
                  extent=extent, aspect="auto", interpolation="nearest")
        displacement = np.sum((repaired[method][middle, cut] - rail[cut]) * direction[cut], axis=-1)
        ax.plot(columns[cut], target_curve, "--", color="#52e5ed", linewidth=1.8, label="Held-out trace (projection)")
        ax.plot(columns[cut], displacement, color="#ffaf44", linewidth=1.4, label="Reconstruction")
        for col in (start - .5, stop - .5):
            ax.axvline(col, color="white", alpha=.55, linewidth=.8)
        ax.set_title(title, loc="left", fontsize=11, pad=6)
        ax.set_ylabel("Normal offset\n(voxels)")
        ax.legend(loc="upper right", fontsize=8)
    if uncertain[middle].any():
        axs[1].scatter(columns[cut][uncertain[middle]], np.full(uncertain[middle].sum(), -44),
                       color="#e779ee", marker="|", s=40, label="Review flag")
        axs[1].legend(loc="upper right", fontsize=8)
    axs[1].set_xlabel(f"Mesh column within crop • Max lateral offset of projected trace = {off_plane.max():.2f} voxels")
    fig.savefig(out / "PHerc0139_sheet_repair.png", dpi=105)
    plt.close(fig)
