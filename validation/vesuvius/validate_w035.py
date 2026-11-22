"""Run two CPU renders of the public PHerc0139 w035 crop and record their results."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
import urllib.request


SEGMENT_URL = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/"
    "20260317000000-w035_2026031718/mesh/"
    "20260317000000-on-20250728140407-9.362um.tifxyz/"
)
VOLUME_URL = (
    "s3://vesuvius-challenge-open-data/PHerc0139/volumes/"
    "20250728140407-9.362um-1.2m-113keV-masked.zarr/"
)
INPUT_FILES = ("meta.json", "x.tif", "y.tif", "z.tif")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(segment):
    segment.mkdir(parents=True, exist_ok=True)

    def get_file(name):
        path = segment / name
        if path.is_file() and path.stat().st_size:
            return
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(SEGMENT_URL + name, timeout=60) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        if not partial.stat().st_size:
            raise ValueError(f"Empty download: {name}")
        partial.replace(path)

    with ThreadPoolExecutor(max_workers=len(INPUT_FILES)) as pool:
        list(pool.map(get_file, INPUT_FILES))


def dimensions(path):
    """Read the first TIFF IFD's spatial dimensions for classic TIFF or BigTIFF."""
    with path.open("rb") as image:
        order = {b"II": "<", b"MM": ">"}.get(image.read(2))
        if order is None:
            raise ValueError(f"Invalid TIFF byte order: {path}")

        def read(fmt):
            return struct.unpack(order + fmt, image.read(struct.calcsize(order + fmt)))

        (version,) = read("H")
        if version == 42:
            offset, count, entry = "I", "H", "HHI4s"
        elif version == 43 and read("HH") == (8, 0):
            offset, count, entry = "Q", "Q", "HHQ8s"
        else:
            raise ValueError(f"Unsupported TIFF header: {path}")
        image.seek(read(offset)[0])
        tags = {}
        for _ in range(read(count)[0]):
            tag, kind, values, value = read(entry)
            if tag in (256, 257) and values == 1 and kind in (3, 4, 16):
                tags[tag] = struct.unpack_from(order + {3: "H", 4: "I", 16: "Q"}[kind], value)[0]
        return [tags[256], tags[257]]


def run(renderer, segment, results, volume, count, timeout):
    label = f"slices-{count}"
    output = results / label
    cmd = [
        str(renderer), "--volume", volume, "--segmentation", str(segment),
        "--scale", "1", "--group-idx", "0", "--cache-gb", "1", "--flip-normals",
        "--crop-x", "2750", "--crop-y", "2450", "--crop-width", "128",
        "--crop-height", "128", "--num-slices", str(count), "--tif-output", str(output),
    ]
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    with (results / f"{label}.stdout.log").open("wb") as stdout:
        with (results / f"{label}.stderr.log").open("wb") as stderr:
            try:
                code = subprocess.run(
                    cmd, cwd=results, stdout=stdout, stderr=stderr, timeout=timeout
                ).returncode
            except subprocess.TimeoutExpired:
                code = None
    elapsed = round(time.monotonic() - start, 3)
    files = sorted(path for path in output.rglob("*") if path.suffix.lower() in (".tif", ".tiff"))
    tiffs = []
    for path in files:
        info = {"file": str(path.relative_to(results)), "sha256": sha256(path)}
        try:
            info["size_xy"] = dimensions(path)
        except (KeyError, OSError, ValueError, struct.error) as error:
            info["header_error"] = str(error)
        tiffs.append(info)
    return {
        "num_slices": count,
        "started_utc": started,
        "command": subprocess.list2cmdline(cmd),
        "elapsed_seconds": elapsed,
        "exit_code": code,
        "timed_out": code is None,
        "output_directory": str(output),
        "output_exists": output.is_dir(),
        "tiff_count": len(tiffs),
        "tiffs": tiffs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("renderer", type=Path)
    parser.add_argument("segment", type=Path)
    parser.add_argument("results", type=Path, help="path for a new results directory")
    parser.add_argument("--download", action="store_true", help="download missing w035 input files")
    parser.add_argument("--volume", default=VOLUME_URL)
    parser.add_argument("--timeout", type=float, default=120, help="seconds allowed per trial")
    args = parser.parse_args()
    renderer = args.renderer.resolve(strict=True)
    segment = args.segment.resolve()
    results = args.results.resolve()
    results.mkdir(parents=True)

    if args.download:
        try:
            download(segment)
        except (OSError, ValueError) as error:
            (results / "download-error.log").write_text(str(error) + "\n", encoding="utf-8")
            raise SystemExit(str(error)) from error
    inputs = {}
    for name in INPUT_FILES:
        candidate = segment / name
        inputs[name] = sha256(candidate) if candidate.is_file() else None
    record = {
        "renderer": str(renderer),
        "renderer_sha256": sha256(renderer),
        "segment_source": SEGMENT_URL,
        "segment": str(segment),
        "input_sha256": inputs,
        "volume": args.volume,
        "trials": [],
    }

    for count in (1, 0):
        trial = run(renderer, segment, results, args.volume, count, args.timeout)
        record["trials"].append(trial)
        (results / "results.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"num-slices={count}, exit={trial['exit_code']}, "
            f"output={trial['output_exists']}, TIFFs={trial['tiff_count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
