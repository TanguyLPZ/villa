"""Compare the Windows tools on the public w035, w047 and w048 papyrus meshes."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

from validate_w035 import INPUT_FILES, SEGMENT_URL, VOLUME_URL, run, sha256


FIRST_LETTERS = (
    "https://raw.githubusercontent.com/Nieuwlaar/pherc343-first-letters/"
    "e8370f1f88a211166056df62ea6fcfffe4a1063e/outputs/"
)
REFERENCE_W035_PIXELS = "497339d3587a664e62e6ba8ed778164c2873dc615a1e7d2d4110333faf7b67f3"
FIXTURES = {
    "w035": (SEGMENT_URL, (
        "11752e63185cc6bc8842cf6f4d7aef84a7e3ec7b0b4eb5c10bf99842e7cd8dda",
        "d827a5d2e51035c91da74b5fba4fea55c9df9d8f58c6ec7f534d6012ee2f7b69",
        "41f1df324572b68450f4da92dcd4598844f602982f1965e5bea67e6de080d664",
        "b61f8bbcd9cb538f43cd72678fd4d38a25cc1643daa4bcfdd7f2e9954f9a1e81",
    )),
    "w047": (FIRST_LETTERS + "w047_R5B2_z9500-11000.tifxyz/", (
        "2cfc8dbc5356131721545d63b9d0e460c23501589a79c1b5a4f5755cb4e86643",
        "29117616a3174eaf58bf84c0e36df27089d743d5ede184fcba1d1d9c73034e66",
        "505bea6a149b0a9f3b0a291b1544aade93d50883cfd86c9ec5153ed2b9666a8e",
        "0a9ef23fc0c2a24717aa099ec20659f21444f5153096d04fb295789f64ffa141",
    )),
    "w048": (FIRST_LETTERS + "w048_R5B2_z9500-11000.tifxyz/", (
        "e8cf5c663b185e6db7941eeedb894cdf205ce4ed51a5ce369d677b3dc9ca87d9",
        "4beb95abe7014146aebb58aca9c1e0a7e41a3d5fcd5c30dfd1a1d11b2b1413ec",
        "306f3374db29c41793d5d77bbd28a3e3bf78bf6c3f5f7c09b14d0efdb75722cb",
        "173d5040e20ae8c0106a152d70b7aa005463df54a4ce2b7d2abfea93d1719cec",
    )),
}
EXPECTED = {
    "w035": (62913, 63499, 62329, 62913, 125826),
    "w047": (22435, 22886, 21986, 22435, 44870),
    "w048": (23158, 23623, 22695, 23158, 46316),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def download_verified(url, target, digest):
    if not target.is_file():
        partial = target.with_suffix(target.suffix + ".part")
        for attempt in range(3):
            try:
                with urlopen(url, timeout=45) as response, partial.open("wb") as file:
                    shutil.copyfileobj(response, file)
                require(sha256(partial) == digest, f"Unexpected fixture SHA-256: {url}")
                partial.replace(target)
                break
            except URLError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    require(sha256(target) == digest, f"Unexpected fixture SHA-256: {target}")


def prepare_inputs(root):
    inputs = {}
    for sample, (url, digests) in FIXTURES.items():
        directory = root / f"{sample}.tifxyz"
        directory.mkdir(parents=True, exist_ok=True)
        for file, digest in zip(INPUT_FILES, digests, strict=True):
            download_verified(url + file, directory / file, digest)
        inputs[sample] = directory
    return inputs


def command(executable, *args):
    argv = [str(executable), *map(str, args)]
    print(subprocess.list2cmdline(argv), flush=True)
    subprocess.run(argv, check=True, timeout=150)


def tiff_pixels(file, mode):
    from PIL import Image

    with Image.open(file) as image:
        require(image.mode == mode, f"Unexpected {file} TIFF mode: {image.mode}")
        return image.size, image.tobytes() if mode == "L" else list(image.getdata())


def render_cases(bin_dir, data, root, phase):
    results = root / phase / "render"
    results.mkdir(parents=True)
    renderer = bin_dir / "vc_render_tifxyz.exe"
    trials = {}
    for count in (1, 0, -1):
        trial = run(renderer, data, results, VOLUME_URL, count, 120)
        label = f"slices-{count}"
        stdout = (results / f"{label}.stdout.log").read_text(encoding="utf-8", errors="replace")
        stderr = (results / f"{label}.stderr.log").read_text(encoding="utf-8", errors="replace")
        print(json.dumps(trial, ensure_ascii=True), flush=True)
        print(stdout, stderr, flush=True)
        opened_remote = "remote zarr streaming" in (stdout + stderr).casefold()
        if count == 1:
            require(trial["exit_code"] == 0 and trial["tiff_count"] == 1, "Expected one successful TIFF")
            result_path = results / trial["tiffs"][0]["file"]
            size, pixels = tiff_pixels(result_path, "L")
            require(size == (128, 128), f"Unexpected render shape: {size}")
            trial["decoded_pixel_sha256"] = hashlib.sha256(pixels).hexdigest()
        elif phase == "after":
            require(trial["exit_code"] == 1 and not trial["output_exists"] and not opened_remote,
                    f"Invalid count {count} started I/O")
            require("num-slices must be positive" in stderr, f"Missing diagnostic for count {count}")
        else:
            require(trial["exit_code"] == 1 and trial["tiff_count"] == 0 and trial["output_exists"],
                    f"Original behavior not reproduced for {count}")
            require(opened_remote, f"Original renderer did not open S3 for {count}")
        trials[str(count)] = {
            **trial, "opened_remote": opened_remote,
            "diagnostic": stderr.strip().splitlines()[-1] if stderr.strip() else None,
        }
    return trials


def read_grid(directory):
    grids = [tiff_pixels(directory / file, "F") for file in INPUT_FILES[1:]]
    width, height = grids[0][0]
    require(all(shape == (width, height) for shape, _ in grids), f"Invalid shape in {directory}")
    coords = list(zip(*(pixels for _, pixels in grids), strict=True))
    points = {
        pixel: xyz for pixel, xyz in enumerate(coords)
        if xyz[2] > 0 and all(math.isfinite(v) and v != -1 for v in xyz)
    }
    quads = sum(
        all(i in points for i in (corner, corner + 1, corner + width, corner + width + 1))
        for row in range(height - 1)
        for corner in range(row * width, (row + 1) * width - 1)
    )
    return (width, height), points, quads


def obj_faces(file):
    vertices = []
    triangles = Counter()
    with file.open(encoding="utf-8") as obj:
        for line in obj:
            if line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                indices = [int(vertex.split("/", 1)[0]) - 1 for vertex in line.split()[1:]]
                require(len(indices) == 3, f"Expected OBJ triangles: {file}")
                triangles[tuple(sorted(vertices[index] for index in indices))] += 1
    return triangles


def mesh_cases(bin_dir, inputs, root, phase):
    exporter = bin_dir / "vc_tifxyz2obj.exe"
    importer = bin_dir / "vc_obj2tifxyz.exe"
    samples = {}
    for sample, source in inputs.items():
        folder = root / phase / sample
        folder.mkdir()
        mesh = root / "before" / sample / f"{sample}.obj"
        if phase == "before":
            command(exporter, source, mesh)
        converted = folder / f"{sample}-roundtrip.tifxyz"
        command(importer, mesh, converted, f"--tifxyz-source={source}")
        shape, points, quads = read_grid(converted)
        b_points, a_points, b_quads, a_quads, total_faces = EXPECTED[sample]
        expected = (b_points, b_quads) if phase == "before" else (a_points, a_quads)
        require((len(points), quads) == expected, f"Unexpected {sample} counts: {len(points)}, {quads}")
        stats = {"grid_xy": shape, "points": len(points), "quads": quads}
        if phase == "after":
            original = root / "before" / sample / converted.name
            before_shape, before_points, _ = read_grid(original)
            require(before_shape == shape, f"{sample} grid dimensions changed")
            require(all(points.get(pixel) == xyz for pixel, xyz in before_points.items()),
                    f"{sample} lost or moved a previously valid point")
            reexport = folder / f"{sample}-reexport.obj"
            command(exporter, converted, reexport)
            triangles = obj_faces(reexport)
            require(triangles == obj_faces(mesh) and triangles.total() == total_faces,
                    f"{sample} OBJ triangle geometry changed")
            stats["original_triangles_preserved"] = triangles.total()
            stats["recovered_points"] = len(points) - len(before_points)
        samples[sample] = stats
        print(f"{phase} {sample} {json.dumps(stats)}", flush=True)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin_dir", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve(strict=True)
    root = args.results.resolve()
    phase = args.phase
    inputs = prepare_inputs(root / "inputs")
    os.environ["VC3D_CONFIG_DIR"] = str(root / ".vc3d")
    command(bin_dir / "vc_cache_root.exe", "--set", root / "cache")
    render = render_cases(bin_dir, inputs["w035"], root, phase)
    meshes = mesh_cases(bin_dir, inputs, root, phase)
    record = {"render": render, "meshes": meshes, "phase": phase}

    if phase == "after":
        before = json.loads((root / "before.json").read_text(encoding="utf-8"))
        original_pixels = before["render"]["1"]["decoded_pixel_sha256"]
        patched_pixels = render["1"]["decoded_pixel_sha256"]
        require(original_pixels == patched_pixels, "The patch changed w035 pixels")
        record["render_pixels_match_official_reference"] = patched_pixels == REFERENCE_W035_PIXELS
        record["baseline_commit"] = os.environ.get("BASE_COMMIT")
        record["tested_commit"] = os.environ.get("GITHUB_SHA")

    (root / f"{phase}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("VC3D_EVIDENCE_JSON " + json.dumps(record, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
