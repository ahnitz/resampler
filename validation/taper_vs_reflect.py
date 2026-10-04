#!/usr/bin/env python3
"""Does shortening the kernel near a science edge beat reflecting?

The idea is appealing: an output sample d samples from an edge can only draw
on d samples of real data, so use the longest filter that FITS and never
invent input at all.  `design_ladder` builds that ladder.

Measured, it loses.  Reflection's error enters only through the extreme tail
of the kernel and so decays very fast with distance, while a shortened kernel
has a systematically different response that applies at full strength
everywhere it is used.  Perturbing the input slightly costs far less than
changing the filter.

Comparison is restricted to 20-900 Hz, strictly inside every ladder passband,
so the shortened filters are not penalised for the band they openly drop.
"""
import os
import sys

import h5py
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_sampler.filters import _apply_ladder, design, design_ladder, filter_block  # noqa: E402

FS = 16384
WINDOWS = [(600, 1000), (1200, 1900), (2200, 3900)]


def main(path, rate=2048, n_edges=10):
    with h5py.File(path, "r") as fh:
        x = fh["strain/Strain"][:1500 * FS].astype(float)
    taps, _ = design(rate)
    ladder = design_ladder(rate)
    truth = signal.oaconvolve(x, taps, mode="same")
    print(f"{'window from edge':>26} {'reflect':>11} {'taper':>11}  winner")
    for lo, hi in WINDOWS:
        er, et = [], []
        for cut in range(200 * FS, (200 + 100 * n_edges) * FS, 100 * FS):
            seg, tr = x[cut:cut + 60 * FS], truth[cut:cut + 60 * FS]
            refl = filter_block(seg, taps)
            tap, _ = _apply_ladder(seg, ladder, FS, FS // rate)
            w = np.hanning(hi - lo)
            f = np.fft.rfftfreq(hi - lo, 1 / FS)
            T = np.fft.rfft(tr[lo:hi] * w)
            R = np.fft.rfft(refl[lo:hi] * w)
            P = np.fft.rfft(tap[lo:hi] * w)
            b = (f > 20) & (f < 900)
            den = np.abs(T[b]).mean()
            er.append(np.abs(R[b] - T[b]).mean() / den)
            et.append(np.abs(P[b] - T[b]).mean() / den)
        mr, mt = np.mean(er), np.mean(et)
        print(f"{lo:6d}-{hi:<6d}({lo/FS*1e3:5.1f}-{hi/FS*1e3:5.1f} ms) "
              f"{mr:11.3e} {mt:11.3e}  {'taper' if mt < mr else 'reflect'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: taper_vs_reflect.py <a 16KHZ strain file>")
    main(sys.argv[1])
