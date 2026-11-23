"""Public inputs for the PHerc. 139 surface repair benchmark."""

import concurrent.futures
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SCAN = "20260102150214"
VOXEL_UM = 2.399
CT = "PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/0"
MESHES = {
    "w029": "PHerc0139/segments/20250108000004-w029_2025010827/mesh/20250108000004-on-20260102150214-2.399um.tifxyz/",
    "w030": "PHerc0139/segments/20250108000005-w030_2025010818/mesh/20250108000005-on-20260102150214-2.399um.tifxyz/",
    "w031": "PHerc0139/segments/20260130000000-w031_2026013019/mesh/20260130000000-on-20260102150214-2.399um.tifxyz/",
    "w032": "PHerc0139/segments/20260203000000-w032_2026020303/mesh/20260203000000-on-20260102150214-2.399um.tifxyz/",
}
MESH_FILES = ("meta.json", "x.tif", "y.tif", "z.tif")


def fetch(key, *, method="GET", attempts=3):
    url = f"{BUCKET}/{key}"
    for attempt in range(attempts):
        try:
            request = Request(url, method=method, headers={"User-Agent": "vesuvius-sheet-repair/1.0"})
            with urlopen(request, timeout=25) as response:
                return response.read(), dict(response.headers.items())
        except HTTPError as exc:
            if exc.code < 500 or attempt + 1 == attempts:
                raise
        except (URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise


def metadata():
    for filename in (".zarray", "zarr.json"):
        try:
            response, _ = fetch(f"{CT}/{filename}")
        except HTTPError as exc:
            if exc.code in (403, 404):
                continue
            raise
        return json.loads(response)
    raise RuntimeError("No Zarr array metadata at the published CT level")


def probe():
    info = metadata()
    print("REPAIR_CT " + json.dumps({"url": f"{BUCKET}/{CT}", **info}), flush=True)
    tasks = [(mesh, filename, prefix + filename) for mesh, prefix in MESHES.items() for filename in MESH_FILES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checks = [pool.submit(fetch, key, method="HEAD") for _, _, key in tasks]
        for (mesh, filename, _), check in zip(tasks, checks, strict=True):
            _, headers = check.result()
            print(f"REPAIR_INPUT {mesh} {filename} {headers.get('Content-Length')}", flush=True)


if __name__ == "__main__":
    probe()
