"""Compare a real TIFXYZ export in triangle and quad OBJ encodings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_quads(source, dest):
    count = 0
    first = None
    with source.open(encoding="ascii") as src, dest.open("w", encoding="ascii", newline="\n") as dst:
        for line in src:
            if not line.startswith("f "):
                if first is not None:
                    raise ValueError("Unexpected statements between exported triangles")
                dst.write(line)
                continue
            corners = line.split()[1:]
            if len(corners) != 3:
                raise ValueError("Expected a triangulated VC3D export")
            if first is None:
                first = corners
            else:
                if first[0] != corners[0] or first[2] != corners[1]:
                    raise ValueError("Triangles did not share their original grid diagonal")
                dst.write("f " + " ".join(first + [corners[2]]) + "\n")
                first = None
                count += 1
    if first is not None:
        raise ValueError("Unpaired triangle")
    return count


def run(cmd, log):
    start = time.monotonic()
    result = subprocess.run([str(word) for word in cmd], cwd=log.parent, capture_output=True, timeout=180)
    output = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode("utf-8", errors="replace")
    log.write_text(output, encoding="utf-8")
    record = {"command": [str(word) for word in cmd], "cwd": str(log.parent), "exit": result.returncode,
              "seconds": round(time.monotonic() - start, 3)}
    valid = re.search(r"Valid grid points:\s*(\d+)\s*/\s*(\d+)", output)
    if valid:
        record["valid"], record["total"] = map(int, valid.groups())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin", type=Path)
    parser.add_argument("segment", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.bin, args.segment, args.out = [p.resolve() for p in (args.bin, args.segment, args.out)]
    args.out.mkdir(exist_ok=True, parents=True)
    importer = next(p for name in ("vc_obj2tifxyz", "vc_obj2tifxyz.exe") if (p := args.bin / name).is_file())
    exporter = next(p for name in ("vc_tifxyz2obj", "vc_tifxyz2obj.exe") if (p := args.bin / name).is_file())
    tri, quad = args.out / "triangles.obj", args.out / "quads.obj"
    results = {"input": str(args.segment), "source_sha256": {p.name: digest(p) for p in sorted(args.segment.iterdir()) if p.is_file()},
               "importer_sha256": digest(importer), "exporter_sha256": digest(exporter)}
    if any(args.out.iterdir()):
        parser.error(f"Choose an empty output directory: {args.out}")
    results["export"] = run([exporter, args.segment, tri.name, "--no-inpaint"], args.out / "export.log")
    if results["export"]["exit"]:
        raise RuntimeError((args.out / "export.log").read_text(encoding="utf-8"))
    results["quads"] = write_quads(tri, quad)
    for density, opts in (("native", [f"--tifxyz-source={args.segment}"]), ("four", ["--uv-downsample=5"])):
        for mesh in (tri, quad):
            key = density + "-" + mesh.stem
            dest = args.out / (key + ".tifxyz")
            if dest.exists():
                raise ValueError(f"Choose a fresh output directory: {dest}")
            record = run([importer, mesh.name, dest.name, *opts], args.out / (key + ".log"))
            record["axes_sha256"] = {p.name: digest(p) for p in sorted(dest.glob("*.tif"))}
            results[key] = record
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v if k == "quads" else {x: v.get(x) for x in ("exit", "valid", "total")}
                      for k, v in results.items() if k == "quads" or k.startswith(("native-", "four-"))}))
    for name in ("native-triangles", "native-quads", "four-triangles", "four-quads"):
        result = results[name]
        if result["exit"] or "valid" not in result or len(result["axes_sha256"]) != 3:
            raise SystemExit(f"Incomplete conversion: {args.out / (name + '.log')}")


if __name__ == "__main__":
    main()
