"""Compare Lasagna Offset weights on public PHerc. 139 winding references."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
from time import monotonic
from urllib.error import HTTPError
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from scipy.spatial import cKDTree
import tifffile

from offset_inputs import MESHES, MESH_FILES, field_manifest, request, url


ROOT = Path(__file__).resolve().parents[2]
WINDOW = (10, 20)
CASES = {
    "forward": {"start": "w030", "opposite": "w029", "target": "w031", "beyond": "w032", "column_fraction": 0.5},
    "reverse": {"start": "w031", "opposite": "w032", "target": "w030", "beyond": "w029", "column_fraction": 0.3},
}
CANDIDATES = ("shipped", "offset-dominant", "station-in-refinement", "offset-snap-pr2418")
SHAPE_CANDIDATES = {"offset-with-smooth": ("smooth", 40.0), "offset-with-bend": ("bend", 400.0)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def download(root):
    jobs = []
    for name, prefix in MESHES.items():
        folder = root / f"{name}.tifxyz"
        folder.mkdir(exist_ok=True)
        for filename in (*MESH_FILES, "mask.tif"):
            jobs.append((prefix + filename, folder / filename))

    def fetch(job):
        key, dest = job
        if dest.exists():
            return
        try:
            with request(url(key)) as source:
                payload = source.read()
        except HTTPError as exc:
            if dest.name == "mask.tif" and exc.code in (403, 404):
                return
            raise
        dest.write_bytes(payload)
        print(f"OFFSET_SOURCE {url(key)} sha256={hashlib.sha256(payload).hexdigest()}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, jobs))


def read_surface(folder):
    xyz = np.stack([tifffile.imread(folder / f"{axis}.tif") for axis in "xyz"], axis=-1)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz > 0).all(axis=-1)
    mask = folder / "mask.tif"
    if mask.exists():
        valid &= tifffile.imread(mask) > 0
    return xyz, valid


def projections(xyz):
    normals = np.cross(np.gradient(xyz, axis=0), np.gradient(xyz, axis=1))
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(length, 1e-8)


def select_windows(xyz, valid, references, case):
    """Choose one interior window per row band using traced geometry alone."""
    height, width = WINDOW
    rows, cols = valid.shape
    center_col = (cols - width) * case["column_fraction"]
    bands = np.linspace(0, rows, 4, dtype=int)
    for band, (lo, hi) in enumerate(zip(bands[:-1], bands[1:]), 1):
        candidates = [(row, col) for row in range(lo + 4, hi - height - 4, 12)
                      for col in range(4, cols - width - 4, 12)]
        candidates.sort(key=lambda p: (p[0] - (lo + hi - height) / 2) ** 2
                        + (p[1] - center_col) ** 2)
        for row, col in candidates:
            index = np.s_[row:row + height, col:col + width]
            if not valid[index].all():
                continue
            points = xyz[index]
            flat = points.reshape(-1, 3)
            normals = projections(points).reshape(-1, 3)
            offsets = {}
            for role in ("opposite", "target", "beyond"):
                tree = references[case[role]]
                dist, nearest = tree.query(flat)
                delta = tree.data[nearest] - flat
                offsets[role] = (delta * normals).sum(axis=-1)
                if role == "target":
                    target_distance = dist
            sign = 1 if np.median(offsets["target"]) > 0 else -1
            target = sign * offsets["target"]
            opposite = -sign * offsets["opposite"]
            beyond = sign * offsets["beyond"]
            if (np.quantile(target, 0.1) >= 16
                    and np.quantile(opposite, 0.1) >= 12
                    and np.quantile(beyond - target, 0.1) >= 8
                    and np.quantile(target_distance, 0.9) <= 150):
                yield band, row, col, points, sign, float(np.median(target_distance))
                break
        else:
            print(f"OFFSET_NO_WINDOW band={band} rows={lo}:{hi}", flush=True)


def write_window(folder, xyz, scale, base_shape, source, row, col):
    folder.mkdir(exist_ok=True)
    for i, axis in enumerate("xyz"):
        tifffile.imwrite(folder / f"{axis}.tif", xyz[..., i].astype(np.float32))
    write_json(folder / "meta.json", {
        "uuid": str(uuid5(NAMESPACE_URL, url(MESHES[source]) + f"#{row},{col}")),
        "name": folder.name, "type": "seg", "format": "tifxyz", "scale": scale,
        "base_shape_zyx": base_shape,
        "bbox": [xyz.min(axis=(0, 1)).tolist(), xyz.max(axis=(0, 1)).tolist()],
    })


def score(xyz, start, sign, references, target):
    if xyz.shape != start.shape or not np.isfinite(xyz).all() or not (xyz > 0).all():
        raise ValueError(f"Expected a finite fitted grid matching {start.shape}, got {xyz.shape}")
    flat = xyz.reshape(-1, 3)
    names = list(references)
    distances = np.stack([references[name].query(flat)[0] for name in names])
    winners = np.argmin(distances, axis=0)
    start_normals = projections(start)
    movement = (xyz - start) * start_normals
    opposed_normals = (projections(xyz) * start_normals).sum(axis=-1) < 0
    opposed_triangles = 0
    corner = np.s_[:-1, :-1]
    diagonal = np.s_[1:, 1:]
    for edges in ((np.s_[1:, :-1], diagonal), (diagonal, np.s_[:-1, 1:])):
        before = np.cross(start[edges[0]] - start[corner], start[edges[1]] - start[corner])
        after = np.cross(xyz[edges[0]] - xyz[corner], xyz[edges[1]] - xyz[corner])
        opposed_triangles += int(((before * after).sum(axis=-1) < 0).sum())
    return {
        "vertices": len(flat),
        "nearest_counts": {name: int((winners == i).sum()) for i, name in enumerate(names)},
        "median_distance_to_target": round(float(np.median(distances[names.index(target)])), 3),
        "median_move_toward_target": round(float(np.median(sign * movement.sum(axis=-1))), 3),
        "max_displacement": round(float(np.max(np.linalg.norm(xyz - start, axis=-1))), 3),
        "opposed_vertex_normals": int(opposed_normals.sum()),
        "opposed_triangles": opposed_triangles,
    }


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    download(root)
    refs = {}
    for name in MESHES:
        xyz, valid = read_surface(root / f"{name}.tifxyz")
        refs[name] = cKDTree(xyz[valid])
        print(f"OFFSET_MESH {name} shape={xyz.shape} usable={int(valid.sum())}", flush=True)

    key, field = field_manifest()
    manifest = root / PurePosixPath(key).name
    write_json(manifest, field)
    windows = []
    for direction, case in CASES.items():
        source = root / f'{case["start"]}.tifxyz'
        meta = json.loads((source / "meta.json").read_text())
        start_xyz, start_valid = read_surface(source)
        for band, row, col, xyz, sign, target_distance in select_windows(start_xyz, start_valid, refs, case):
            name = f"{direction}-{band}"
            folder = root / f"{name}.tifxyz"
            write_window(folder, xyz, meta["scale"], field["base_shape_zyx"], case["start"], row, col)
            # The two field scales are 8 and 16. Align the dense crop to both.
            low = 16 * np.floor((xyz.min(axis=(0, 1)) - 256) / 16).astype(int)
            high = 16 * np.ceil((xyz.max(axis=(0, 1)) + 256) / 16).astype(int)
            window = {"name": name, "direction": direction, "band": band, "row": row, "col": col,
                      "mesh": folder.name, "crop_xyzwhd": [*low.tolist(), *(high - low).tolist()],
                      "offset": sign, "start_distance_to_target": round(target_distance, 3)}
            control = score(xyz, xyz, sign, refs, case["target"])
            if control["nearest_counts"][case["start"]] != xyz.shape[0] * xyz.shape[1]:
                raise ValueError(f"Invalid start reference for {window}: {control}")
            window["control"] = control
            print("OFFSET_WINDOW " + json.dumps(window), flush=True)
            windows.append(window)
        if not any(window["direction"] == direction for window in windows):
            raise ValueError(f"No fully traced {case['start']} window with separated nearest references")
    plan = {"field_key": key, "manifest": manifest.name, "windows": windows}
    write_json(root / "plan.json", plan)
    return plan, refs


def fit_window(root, plan, window, name, dense_cache, steps):
    import fit
    import fit_data
    import torch

    manifest = root / plan["manifest"]
    field = json.loads(manifest.read_text())
    groups = {str(root / g["zarr"]): url(str(PurePosixPath(plan["field_key"]).parent / g["zarr"]))
              for g in field["groups"].values()}
    native_open = fit_data.open_zarr
    native_streaming = fit_data.load_3d_streaming
    crop = tuple(window["crop_xyzwhd"])

    def open_public(location, **kwargs):
        return native_open(groups.get(str(location), location), **kwargs)

    def load_cpu_crop(*, path, device, skip_channels=None, require_umbilicus=False):
        key = (path, crop, frozenset(skip_channels or ()), require_umbilicus)
        if key not in dense_cache:
            data = fit_data.load_3d(path=path, device=device, crop=crop, cuda_gridsample=False,
                                   skip_channels=skip_channels, require_umbilicus=require_umbilicus)
            if data.grad_mag is None or not data.grad_mag.any():
                raise ValueError(f"Public grad_mag crop is empty: {crop}")
            dense_cache[key] = data
        return dense_cache[key]

    config_path = (Path(__file__).with_name("offset_snap.pr2418.json") if name == "offset-snap-pr2418"
                   else ROOT / "lasagna/configs/offset.json")
    config = json.loads(config_path.read_text())
    mesh = root / window["mesh"]
    config["external_surfaces"] = [{"path": str(mesh), "offset": window["offset"]}]
    if name in {"offset-dominant", "station-in-refinement", *SHAPE_CANDIDATES}:
        config["base"].update(station_n=0.0, ext_offset=1.0)
    if name == "station-in-refinement":
        config["base"]["station_n"] = 0.001
        config["stages"][0]["w_fac"]["station_n"] = 0.0
    if name in SHAPE_CANDIDATES:
        term, weight = SHAPE_CANDIDATES[name]
        config["base"][term] = weight
    config["voxel_size_um"] = 2.399
    if steps is not None:
        for stage in config["stages"]:
            stage["steps"] = min(steps, stage["steps"])
    out = root / f'{window["name"]}-{name}'
    out.mkdir(exist_ok=True)
    cfg = out / "config.json"
    write_json(cfg, config)
    argv = [str(cfg), "--input", str(manifest), "--model-init", "ext", "--tifxyz-init", str(mesh),
            "--device", "cpu", "--depth", "1", "--cuda-gridsample", "0", "--out-dir", str(out)]
    print("OFFSET_FIT " + shlex.join(["fit.py", *argv]), flush=True)
    torch.manual_seed(0)
    fit_data.open_zarr = open_public
    fit_data.load_3d_streaming = load_cpu_crop
    try:
        status = fit.main(argv)
    finally:
        fit_data.open_zarr = native_open
        fit_data.load_3d_streaming = native_streaming
    if status:
        raise RuntimeError(f"Lasagna exited with {status}")
    checkpoint = torch.load(out / "model_final.pt", map_location="cpu", weights_only=True)
    grid = checkpoint["mesh_flat"]
    if grid.ndim != 4 or tuple(grid.shape[:2]) != (3, 1):
        raise ValueError(f"Unexpected Lasagna checkpoint mesh shape {tuple(grid.shape)}")
    return grid[:, 0].permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--configs", nargs="+", choices=(*CANDIDATES, *SHAPE_CANDIDATES), default=CANDIDATES)
    args = parser.parse_args()
    root = args.work.resolve()
    plan, refs = prepare(root)
    if args.fixtures_only:
        return

    sys.path.insert(0, str(ROOT / "lasagna"))
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print(f"OFFSET_RUNTIME torch={torch.__version__} device=cpu cuda={torch.cuda.is_available()}", flush=True)
    results = []
    for window in plan["windows"]:
        start, _ = read_surface(root / window["mesh"])
        case = CASES[window["direction"]]
        cache = {}
        rows = [{"window": window["name"], "target": case["target"], "config": "start", **window["control"]}]
        for name in args.configs:
            before = monotonic()
            grid = fit_window(root, plan, window, name, cache, args.smoke_steps)
            result = score(grid, start, window["offset"], refs, case["target"])
            result["wall_seconds"] = round(monotonic() - before, 2)
            rows.append({"window": window["name"], "target": case["target"], "config": name, **result})
            print("OFFSET_RESULT " + json.dumps(rows[-1]), flush=True)
        results.extend(rows)
        write_json(root / "results.json", {"plan": plan, "smoke_steps": args.smoke_steps,
                                          "results": results})

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        lines = ["## Lasagna Offset on PHerc. 139", "",
                 "Assignment uses the nearest vertex among four published reference surfaces.", "",
                 "| ROI | Target | Config | w029 | w030 | w031 | w032 | Median distance to target (voxels) | Opposed normals | Opposed triangles |",
                 "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in results:
            counts = [str(r["nearest_counts"][name]) for name in MESHES]
            lines.append("| " + " | ".join([r["window"], r["target"], r["config"], *counts,
                                          str(r["median_distance_to_target"]), str(r["opposed_vertex_normals"]),
                                          str(r["opposed_triangles"])]) + " |")
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
