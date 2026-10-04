#!/usr/bin/env python3
"""Compare padding strategies at a science-segment edge.

A linear-phase FIR needs ntaps//2 samples either side of every output sample.
At a true science edge those do not exist, so something must be synthesised.
This measures each choice against a reference filtered with full real context.

Result: zero-padding is ~40x worse than anything else, because it asserts an
instantaneous step to zero that the filter then rings on.  Edge-value hold,
even (mirror) reflection and odd reflection are all within a factor of two of
each other, and all are exact beyond ntaps//2 samples from the edge.
"""
import os
import sys

import h5py
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_sampler.filters import design  # noqa: E402

BINS = [(0, 128), (128, 512), (512, 2048), (2048, 4096), (4096, 8192)]
KINDS = ["zero", "const", "even", "odd"]


def pad_filter(seg, kind, taps):
    k = len(taps) // 2
    if kind == "zero":
        left = np.zeros(k); right = np.zeros(k)
    elif kind == "const":
        left = np.full(k, seg[0]); right = np.full(k, seg[-1])
    elif kind == "even":
        left = seg[1:k + 1][::-1]; right = seg[-k - 1:-1][::-1]
    else:
        left = 2 * seg[0] - seg[1:k + 1][::-1]
        right = 2 * seg[-1] - seg[-k - 1:-1][::-1]
    xp = np.concatenate((left, seg, right))
    return signal.oaconvolve(xp, taps, mode="same")[k:k + len(seg)]


def main(path, rate=2048, n_edges=10):
    with h5py.File(path, "r") as f:
        x = f["strain/Strain"][:2000 * 16384].astype(float)
    taps, _ = design(rate)
    full = signal.oaconvolve(x, taps, mode="same")
    rms = x.std()
    acc = {k: [[] for _ in BINS] for k in KINDS}
    for cut in range(200 * 16384, (200 + 100 * n_edges) * 16384, 100 * 16384):
        seg, truth = x[cut:cut + 40 * 16384], full[cut:cut + 40 * 16384]
        for k in KINDS:
            y = pad_filter(seg, k, taps)
            for i, (a, b) in enumerate(BINS):
                acc[k][i].append(
                    np.sqrt(np.mean((y[a:b] - truth[a:b]) ** 2)) / rms)
    print(f"RMS error / data RMS, averaged over {n_edges} segment edges")
    print(f"{'samples from edge':>20} " + " ".join(f"{k:>10}" for k in KINDS))
    for i, (a, b) in enumerate(BINS):
        print(f"{a:7d}-{b:<7d}({b/16384*1e3:5.0f}ms) "
              + " ".join(f"{np.mean(acc[k][i]):10.2e}" for k in KINDS))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: padding_bench.py <a 16KHZ strain file>")
    main(sys.argv[1])
