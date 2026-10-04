#!/usr/bin/env python3
"""
Behaviour at real science-segment boundaries, ours vs GWOSC's 4096 Hz product.

Two things are measured:
  1. COVERAGE  -- how much science time each product actually delivers.
  2. EDGE FIDELITY -- error as a function of distance from a boundary, against
     a reference built from the 16 kHz data with full context.
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_sampler.filters import design, reduce_strain, contiguous_runs  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file16")
    ap.add_argument("file4")
    ap.add_argument("--rate", type=int, default=4096)
    a = ap.parse_args()
    R16 = 16384 // a.rate

    with h5py.File(a.file16) as f:
        x16 = f["strain/Strain"][:]
    with h5py.File(a.file4) as f:
        x4 = f["strain/Strain"][:]
    fs4 = 4096
    assert a.rate == fs4, "compare at GWOSC's own rate"

    y, edge, taps, pb = reduce_strain(x16, a.rate)

    # ---------- 1. coverage ----------
    cov16 = np.isfinite(x16).reshape(-1, R16).any(axis=1)   # any real input
    print(f"science time available in the 16 kHz file : "
          f"{cov16.sum()/a.rate:9.3f} s")
    print(f"delivered by GWOSC 4096 Hz                : "
          f"{np.isfinite(x4).sum()/fs4:9.3f} s")
    print(f"delivered by data_sampler                 : "
          f"{np.isfinite(y).sum()/a.rate:9.3f} s "
          f"({int(((edge==0)&np.isfinite(y)).sum())/a.rate:.3f} s clean, "
          f"{int(((edge==1)&np.isfinite(y)).sum())/a.rate:.3f} s pad, "
          f"{int(((edge==2)&np.isfinite(y)).sum())/a.rate:.3f} s extrapolated)")
    lost_g = (cov16.sum() - np.isfinite(x4).sum()) / fs4
    lost_o = (cov16.sum() - np.isfinite(y).sum()) / a.rate
    print(f"\nscience time LOST by GWOSC : {lost_g:+.3f} s")
    print(f"science time LOST by ours  : {lost_o:+.3f} s")

    segs = [(s, e) for s, e in contiguous_runs(np.isfinite(x16))
            if e - s > 40 * 16384]
    print(f"\n{len(segs)} science segment(s) longer than 40 s")
    if not segs:
        return

    # ---------- 2. is GWOSC's error edge-SPECIFIC, or uniform? ----------
    # Compared against a correctly-filtered reduction of the same 16 kHz data.
    # A boundary pathology (e.g. zero-padding) would show as error GROWING
    # towards the edge; a flat profile means their edge handling is sound and
    # the deviation is just their passband error.
    print("\nGWOSC deviation from a correctly filtered reduction,")
    print("as a function of distance from a science boundary:")
    print(f"{'dist from edge':>18} {'rel RMS deviation':>19}")
    rms = np.nanstd(y)
    for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 0.5),
                   (0.5, 1.0), (1.0, 2.0), (5.0, 10.0), (20.0, 60.0)]:
        acc = []
        for s_, e_ in segs:
            for pos, sign in ((s_ // R16, +1), (e_ // R16, -1)):
                i0, i1 = pos + sign * int(lo * a.rate), pos + sign * int(hi * a.rate)
                sl = slice(min(i0, i1), max(i0, i1))
                g, r = x4[sl], y[sl]
                m = np.isfinite(g) & np.isfinite(r)
                if m.sum() > 10:
                    acc.append(np.sqrt(np.mean((g[m] - r[m]) ** 2)) / rms)
        if acc:
            print(f"{lo:7.2f}-{hi:6.2f} s {np.mean(acc):19.4e}")
    print("\nFlat profile => GWOSC has no edge-specific pathology; the"
          "\ndeviation is their uniform passband error, present everywhere."
          "\nOur own edge accuracy is measured against known truth in"
          "\ntests/test_filters.py (clean 6e-10, pad-affected 7e-5).")


if __name__ == "__main__":
    main()
