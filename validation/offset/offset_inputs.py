"""Locate the public PHerc. 139 fixtures for the Offset CPU benchmark."""

import json
from pathlib import PurePosixPath
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree


BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SCAN = "20260102150214"
FIELD_PREFIX = "PHerc0139/representations/predictions/lasagna/20260102150214-lasagna-20260419180421-L2/"
FIELD_STEM = "PHerc0139-20260102150214-lasagna-20260724"
MESHES = {
    "w029": "PHerc0139/segments/20250108000004-w029_2025010827/mesh/20250108000004-on-20260102150214-2.399um.tifxyz/",
    "w030": "PHerc0139/segments/20250108000005-w030_2025010818/mesh/20250108000005-on-20260102150214-2.399um.tifxyz/",
    "w031": "PHerc0139/segments/20260130000000-w031_2026013019/mesh/20260130000000-on-20260102150214-2.399um.tifxyz/",
    "w032": "PHerc0139/segments/20260203000000-w032_2026020303/mesh/20260203000000-on-20260102150214-2.399um.tifxyz/",
}
MESH_FILES = ("meta.json", "x.tif", "y.tif", "z.tif")


def request(url, method="GET"):
    for attempt in range(3):
        try:
            return urlopen(Request(url, method=method, headers={"User-Agent": "villa-offset-cpu"}), timeout=45)
        except HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise
            if attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def url(key):
    return BUCKET + "/" + quote(key, safe="/")


def public_json(key):
    with request(url(key)) as response:
        return json.load(response)


def field_manifest():
    canonical = FIELD_PREFIX + FIELD_STEM + ".lasagna.json"
    try:
        return canonical, public_json(canonical)
    except HTTPError as exc:
        if exc.code not in (403, 404):
            raise

    query = urlencode({"list-type": 2, "prefix": FIELD_PREFIX, "delimiter": "/", "max-keys": 100})
    with request(BUCKET + "/?" + query) as response:
        listing = ElementTree.fromstring(response.read())
    candidates = [x.text for x in listing.findall("./{*}Contents/{*}Key") if x.text and x.text.endswith(".lasagna.json")]
    print("FIELD_LISTING", json.dumps([PurePosixPath(x.text).name for x in listing.findall("./{*}Contents/{*}Key") if x.text]), flush=True)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Lasagna manifest in {FIELD_PREFIX}, got {candidates}")
    return candidates[0], public_json(candidates[0])


def main():
    meshes = {}
    for segment, prefix in MESHES.items():
        meta = public_json(prefix + "meta.json")
        if meta.get("format") != "tifxyz":
            raise RuntimeError(f"{segment} does not identify a tifxyz surface")
        sizes = {}
        for name in MESH_FILES[1:]:
            with request(url(prefix + name), method="HEAD") as response:
                sizes[name] = int(response.headers["Content-Length"])
        meshes[segment] = {"url": url(prefix), "bbox": meta.get("bbox"), "scale": meta.get("scale"), "bytes": sizes}
        print("MESH", segment, json.dumps(meshes[segment]), flush=True)

    key, doc = field_manifest()
    channels = {ch for g in doc["groups"].values() for ch in g["channels"]}
    if not {"cos", "grad_mag", "nx", "ny"} <= channels:
        raise RuntimeError(f"Field missing required channels: {sorted(channels)}")
    summary = {
        "scan": SCAN, "meshes": meshes, "field_url": url(key), "field_key": key,
        "field_keys": sorted(doc), "source_to_base": doc.get("source_to_base"),
        "base_shape_zyx": doc.get("base_shape_zyx"), "field_groups": doc["groups"],
    }
    print("OFFSET_INPUTS_JSON " + json.dumps(summary), flush=True)
    print("OFFSET_FIELD_DOCUMENT " + json.dumps(doc), flush=True)
    for name, group in doc["groups"].items():
        path = str(PurePosixPath(key).parent / group["zarr"] / ".zarray")
        print("OFFSET_FIELD_ARRAY " + json.dumps({"name": name, "spec": public_json(path)}), flush=True)


if __name__ == "__main__":
    main()
