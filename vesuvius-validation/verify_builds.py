"""Compare two complete converter builds on one public surface (Python 3.11+, Pillow)."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
from reproduce import digest, run, write_quads


def executable(directory, stem):
    for suffix in ("", ".exe"):
        path = directory / (stem + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(directory / stem)


def decoded_axes(directory):
    result = {}
    for axis in "xyz":
        with Image.open(directory / (axis + ".tif")) as image:
            result[axis] = {"size": image.size, "mode": image.mode, "sha256": sha256(image.tobytes()).hexdigest()}
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline_bin", "patched_bin", "segment", "output"):
        cli.add_argument(arg, type=Path)
    args = cli.parse_args()
    before, after, segment, out = [p.resolve() for p in vars(args).values()]
    out.mkdir(parents=True, exist_ok=False)
    tri, quad = out / "triangles.obj", out / "quads.obj"
    exporter = executable(before, "vc_tifxyz2obj")
    export = run([exporter, segment, tri.name, "--no-inpaint"], out / "export.log")
    if export["exit"]:
        raise RuntimeError(out / "export.log")
    report = {"input": segment.name, "input_sha256": {p.name: digest(p) for p in sorted(segment.iterdir()) if p.is_file()},
              "export": export, "exporter_sha256": digest(exporter), "quads": write_quads(tri, quad),
              "binaries": {}, "cases": {}}
    problems = []
    for label, binary, opts in (("native", "vc_obj2tifxyz", [f"--tifxyz-source={segment}"]),
                                ("dense", "vc_obj2tifxyz", ["--uv-downsample=5"]),
                                ("legacy", "vc_obj2tifxyz_legacy", ["5", "--uv-scale=1"])):
        rows = {}
        report["cases"][label] = rows
        for role, root in (("baseline", before), ("patched", after)):
            exe = executable(root, binary)
            report["binaries"][role + "-" + binary] = digest(exe)
            for mesh in (tri, quad):
                key = role + "-" + mesh.stem
                name = label + "-" + key
                dest = out / (name + ".tifxyz")
                record = run([exe, mesh.name, dest.name, *opts], out / (name + ".log"))
                if record["exit"] or "valid" not in record:
                    raise RuntimeError(out / (name + ".log"))
                record["decoded"] = decoded_axes(dest)
                rows[key] = record
        control = rows["baseline-triangles"]
        if control["valid"] <= rows["baseline-quads"]["valid"]:
            problems.append(label + ": the supplied baseline did not reproduce a loss on this input")
        for fixed in ("patched-triangles", "patched-quads"):
            if rows[fixed]["valid"] != control["valid"] or rows[fixed]["decoded"] != control["decoded"]:
                problems.append(label + ": " + fixed + " differs from the triangle control")
        print(label, {key: rec["valid"] for key, rec in rows.items()}, flush=True)
    report["failures"] = problems
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()
