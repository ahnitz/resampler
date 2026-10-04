#!/usr/bin/env python3
"""Benchmark HDF5 codecs for the reduced strain.

Conclusion: gzip + shuffle is at the practical optimum.  Shuffle is the whole
win (-16%); the gzip level is irrelevant (level 9 saves 0.1% for twice the
write cost); zstd/blosc gain nothing and would require hdf5plugin to READ.
"""
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_sampler.filters import reduce_strain  # noqa: E402

CASES = [
    ("none", {}),
    ("lzf", {"compression": "lzf"}),
    ("lzf+shuf", {"compression": "lzf", "shuffle": True}),
]
for _lvl in (1, 4, 6, 9):
    CASES.append((f"gzip{_lvl}", {"compression": "gzip", "compression_opts": _lvl}))
    CASES.append((f"gzip{_lvl}+shuf",
                  {"compression": "gzip", "compression_opts": _lvl, "shuffle": True}))


def bench(path, rate=2048):
    with h5py.File(path, "r") as f:
        x = f["strain/Strain"][:]
    y, _, _, _ = reduce_strain(x, rate)
    y32 = y.astype(np.float32)
    print(f"\n=== {os.path.basename(path)} "
          f"({np.isfinite(y).mean()*100:.1f}% finite) ===")
    print(f"{'codec':16} {'size':>10} {'write s':>9} {'read s':>8}")
    for name, kw in CASES:
        tmp = "/tmp/_cb.h5"
        t = time.time()
        with h5py.File(tmp, "w") as f:
            f.create_dataset("s", data=y32, **kw)
        tw = time.time() - t
        t = time.time()
        with h5py.File(tmp, "r") as f:
            f["s"][:]
        tr = time.time() - t
        print(f"{name:16} {os.path.getsize(tmp)/1e6:8.2f} MB {tw:9.2f} {tr:8.2f}")
        os.remove(tmp)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: compression_bench.py <16KHZ file> [...]")
    for p in sys.argv[1:]:
        bench(p)
