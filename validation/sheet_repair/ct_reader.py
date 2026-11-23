"""Sample the public, uncompressed PHerc. 139 CT without staging the volume."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from urllib.error import HTTPError

import numpy as np

from public_data import CT, fetch, metadata


class CTReader:
    def __init__(self):
        info = metadata()
        required = {"zarr_format": 2, "dtype": "|u1", "order": "C", "filters": None,
                    "dimension_separator": "/", "compressor": None, "fill_value": 0}
        if any(info.get(key) != value for key, value in required.items()):
            raise ValueError(f"Unexpected public CT array encoding: {info}")
        self.shape = np.asarray(info["shape"], dtype=np.int64)
        self.chunk_shape = np.asarray(info["chunks"], dtype=np.int64)
        if self.shape.shape != (3,) or self.chunk_shape.shape != (3,) or (self.chunk_shape <= 0).any():
            raise ValueError(f"Expected a 3-D CT array: {info}")
        self.chunk_grid = tuple((self.shape + self.chunk_shape - 1) // self.chunk_shape)
        self.cache = OrderedDict()
        self.blocks_read = 0
        self.bytes_read = 0

    def _read_block(self, index):
        key = "/".join(str(value) for value in np.unravel_index(index, self.chunk_grid))
        try:
            raw, _ = fetch(f"{CT}/{key}")
        except HTTPError as exc:
            if exc.code != 404:
                raise
            return np.zeros(tuple(self.chunk_shape), dtype=np.uint8), 0
        if len(raw) != int(np.prod(self.chunk_shape)):
            raise ValueError(f"Unexpected byte count for CT chunk {key}: {len(raw)}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.chunk_shape), len(raw)

    def sample(self, xyz):
        """Trilinear grayscale samples at level-zero XYZ coordinates."""
        points = np.asarray(xyz, dtype=np.float32)
        if points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("CT sampling expects finite XYZ points")
        zyx = points.reshape(-1, 3)[:, ::-1]
        low = np.floor(zyx).astype(np.int32)
        if np.any(low < 0) or np.any(low + 1 >= self.shape):
            raise ValueError("Sample coordinates extend outside the CT volume")
        corners = np.asarray(list(product((0, 1), repeat=3)), dtype=np.int32)
        indices = (low[:, None, :] + corners[None, :, :]).reshape(-1, 3)
        keys = np.ravel_multi_index((indices // self.chunk_shape).T, self.chunk_grid)
        order = np.argsort(keys, kind="stable")
        groups = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(keys)]
        needed = keys[order[groups[:-1]]]
        samples = np.empty(len(keys), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=6) as pool:
            downloads = {index: pool.submit(self._read_block, index)
                         for index in needed if index not in self.cache}
            for index, begin, end in zip(needed, groups[:-1], groups[1:], strict=True):
                if index in downloads:
                    block, size = downloads.pop(index).result()
                    self.blocks_read += 1
                    self.bytes_read += size
                else:
                    block = self.cache.pop(index)
                order_slice = order[begin:end]
                coordinates = indices[order_slice] % self.chunk_shape
                samples[order_slice] = block[tuple(coordinates.T)]
                self.cache[index] = block
            while len(self.cache) > 128:
                self.cache.popitem(last=False)

        fraction = (zyx - low).astype(np.float32)
        weights = np.prod(np.where(corners[None, :, :], fraction[:, None, :],
                                   1 - fraction[:, None, :]), axis=-1)
        return np.sum(weights * samples.reshape(-1, 8), axis=1).reshape(points.shape[:-1])
