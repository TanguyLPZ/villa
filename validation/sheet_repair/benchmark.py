"""Benchmark strip repair on withheld vertices on the PHerc. 139 CT."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.spatial import cKDTree
import tifffile

from ct_reader import CTReader
from evidence import write_evidence
from public_data import BUCKET, CT, MESHES, MESH_FILES, SCAN, VOXEL_UM, fetch
from sheet_repair import cubic_from_four_columns, fit_strip, normals, trace_continuous


# Use the six interior locations already traced on the volume, fixed before scoring.
CROPS = {
    "w030": ((218, 696), (692, 696), (1155, 696)),
    "w031": ((218, 420), (676, 420), (1135, 420)),
}
SHAPE = (14, 52)
WIDTHS = (8, 16, 24)
METHODS = ("linear", "cubic", "cubic_4col", "ct_local", "ct_connected", "ct_continuous")
HOLDOUT_COLUMNS = {"w029": .46, "w032": .52}
HOLDOUT_ROWS = (.17, .50, .83)


def log(tag, payload):
    print(f"REPAIR_{tag} " + json.dumps(payload, allow_nan=False), flush=True)


def read_inputs(directory):
    jobs = [(mesh, filename) for mesh in MESHES for filename in MESH_FILES]
    for mesh in MESHES:
        (directory / mesh).mkdir(parents=True, exist_ok=True)

    def stage(job):
        mesh, filename = job
        key = MESHES[mesh] + filename
        path = directory / mesh / filename
        if not path.is_file():
            content, _ = fetch(key)
            path.write_bytes(content)
        return {"url": f"{BUCKET}/{key}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(stage, jobs))
    inputs = {}
    for mesh in MESHES:
        root = directory / mesh
        surface = np.stack([tifffile.imread(root / f"{axis}.tif") for axis in "xyz"], axis=-1)
        surface = surface.astype(np.float32, copy=False)
        valid = np.isfinite(surface).all(axis=-1) & (surface > 0).all(axis=-1)
        inputs[mesh] = (surface, valid, json.loads((root / "meta.json").read_text(encoding="utf-8")))
    return inputs, files


def select_crop(valid, origin, search_radius):
    height, width = SHAPE
    candidates = range(-search_radius, search_radius + 1, 4)
    offsets = [(i, j) for i in candidates for j in candidates]
    offsets.sort(key=lambda pair: (pair[0] ** 2 + pair[1] ** 2, pair))
    for i, j in offsets:
        row, col = origin[0] + i, origin[1] + j
        window = valid[row:row + height, col:col + width]
        if window.shape == SHAPE and window.all():
            return row, col
    raise ValueError(f"No complete interior crop near {origin}, mask shape {valid.shape}")


def crop_origins(inputs, split):
    if split == "core":
        return CROPS
    return {name: [(round((inputs[name][1].shape[0] - SHAPE[0]) * row_fraction),
                    round((inputs[name][1].shape[1] - SHAPE[1]) * column_fraction))
                   for row_fraction in HOLDOUT_ROWS] for name, column_fraction in HOLDOUT_COLUMNS.items()}


def score(prediction, truth, directions, references, name):
    delta = prediction - truth
    absolute_normal = np.abs(np.sum(delta * directions, axis=-1))
    euclidean = np.linalg.norm(delta, axis=-1)
    neighbors = np.stack([tree.query(prediction.reshape(-1, 3))[0] for tree in references.values()])
    wrong_winding = (neighbors.argmin(axis=0) != tuple(references).index(name)).reshape(truth.shape[:2])
    return absolute_normal, euclidean, wrong_winding


def metrics(data):
    normals, euclidean, wrong_winding = data
    return {
        "vertices": len(normals),
        "normal_mae_vox": float(normals.mean()),
        "normal_median_vox": float(np.median(normals)),
        "normal_p90_vox": float(np.percentile(normals, 90)),
        "euclidean_mae_vox": float(euclidean.mean()),
        "wrong_winding": int(wrong_winding.sum()),
    }


def combine(chunks):
    return tuple(np.concatenate([chunk.ravel() for chunk in group]) for group in zip(*chunks, strict=True))


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    inputs, files = read_inputs(args.inputs)
    cases = []
    for name, origins in crop_origins(inputs, args.split).items():
        for number, origin in enumerate(origins, 1):
            row, col = select_crop(inputs[name][1], origin, 12 if args.split == "core" else 80)
            cases.append({"id": f"{name}-{number}", "mesh": name, "row": row, "col": col})
    log("PLAN", {"split": args.split, "cases": cases, "shape": SHAPE, "widths": WIDTHS, "voxel_um": VOXEL_UM})

    # Full references, including the strip, are used only to score completed predictions.
    references = {name: cKDTree(surface[valid]) for name, (surface, valid, _) in inputs.items()}
    reader = CTReader()
    measurements = {width: {method: [] for method in METHODS} for width in WIDTHS}
    records = []
    review_counts = {method: np.zeros(4, dtype=np.int64) for method in ("ct_connected", "ct_continuous")}
    started = monotonic()
    for case in cases:
        name, row, col = case["mesh"], case["row"], case["col"]
        surface, _, meta = inputs[name]
        truth = surface[row:row + SHAPE[0], col:col + SHAPE[1]].copy()
        direction = normals(truth)
        for width in WIDTHS:
            start = (SHAPE[1] - width) // 2
            stop = start + width
            hidden = np.s_[:, start:stop]
            observed = truth.copy()
            observed[hidden] = np.nan
            before = monotonic()
            repaired, connected_review = fit_strip(observed, start, stop, reader.sample)
            repaired["cubic_4col"] = cubic_from_four_columns(observed, start, stop)
            repaired["ct_continuous"], continuous_review = trace_continuous(repaired["cubic"], start, stop, reader.sample)
            seconds = monotonic() - before
            scores = {}
            errors = {}
            for method in METHODS:
                prediction = repaired[method]
                for retained in (np.s_[:, :start], np.s_[:, stop:]):
                    if not np.array_equal(prediction[retained], observed[retained]):
                        raise ValueError(f"{method} changed a visible vertex")
                result = score(prediction[hidden], truth[hidden], direction[hidden], references, name)
                measurements[width][method].append(result)
                scores[method] = metrics(tuple(item.ravel() for item in result))
                errors[method] = result[0]
            for method, flag in (("ct_connected", connected_review), ("ct_continuous", continuous_review)):
                large_error = errors[method] > 4
                review_counts[method] += (int((~flag).sum()), int((large_error & ~flag).sum()),
                                          int(flag.sum()), int((large_error & flag).sum()))
            record = {**case, "split": args.split, "width": width, "seconds_with_shared_ct_cache": seconds,
                      "review_vertices": {"ct_connected": int(connected_review.sum()),
                                          "ct_continuous": int(continuous_review.sum())}, "methods": scores}
            records.append(record)
            log("CASE", record)
            if case == cases[0] and width == WIDTHS[-1]:
                write_evidence(args.out, reader.sample, case, meta, start, stop, truth, repaired, continuous_review)

    per_width = {width: {method: metrics(combine(chunks)) for method, chunks in by_method.items()}
                 for width, by_method in measurements.items()}
    pooled = {method: metrics(combine([chunk for by_method in measurements.values() for chunk in by_method[method]]))
              for method in METHODS}
    review_headers = ("accepted_count", "accepted_over_threshold", "flagged_count", "flagged_over_threshold")
    review = {"threshold_normal_error_vox": 4,
              **{method: dict(zip(review_headers, counts.tolist(), strict=True))
                 for method, counts in review_counts.items()}}
    stats = {"ct_url": f"{BUCKET}/{CT}", "scan": SCAN, "voxel_um": VOXEL_UM,
             "git_sha": os.environ.get("GITHUB_SHA"), "sources": files, "shape": SHAPE, "split": args.split,
             "cases": records, "by_width": per_width, "pooled": pooled, "review": review,
             "downloaded_ct_chunks": reader.blocks_read, "downloaded_ct_bytes": reader.bytes_read,
             "seconds_with_ct_io_and_plot": monotonic() - started}
    (args.out / "results.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    log("POOLED", pooled)
    log("WIDTHS", per_width)
    log("REVIEW", review)
    log("RUN", {key: stats[key] for key in ("git_sha", "downloaded_ct_chunks", "downloaded_ct_bytes",
                                             "seconds_with_ct_io_and_plot")})
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(f"## PHerc. 139 withheld-strip benchmark ({args.split})\n\n")
            summary.write("| Method | Mean absolute normal error (voxels) | P90 (voxels) | Nearest winding differs |\n")
            summary.write("| --- | ---: | ---: | ---: |\n")
            for method, values in pooled.items():
                summary.write(f"| {method} | {values['normal_mae_vox']:.3f} | {values['normal_p90_vox']:.3f} "
                              f"| {values['wrong_winding']} / {values['vertices']} |\n")
    if args.emit:
        archive = args.out.parent / f"vesuvius-sheet-repair-{args.split}-evidence.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
            for path in sorted(args.out.rglob("*")):
                if path.is_file():
                    output.write(path, arcname=path.relative_to(args.out))
        payload = archive.read_bytes()
        log("ZIP", {"split": args.split, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        encoded = base64.b64encode(payload).decode("ascii")
        for offset in range(0, len(encoded), 3000):
            print(f"REPAIR_DATA {args.split} {offset:08d} {encoded[offset:offset + 3000]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", choices=("core", "holdout"), default="core")
    parser.add_argument("--emit", action="store_true", help="Include the evidence archive in CI logs")
    run(parser.parse_args())
