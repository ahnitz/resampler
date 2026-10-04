#!/usr/bin/env python3
"""Identify the filter GWOSC used for its 4096 Hz product, and verify it.

GWOSC documents (https://gwosc.org/O4/o4_details/) that the down-sampling is
done "using the python package scipy, with the method scipy.signal.decimate",
and warns that "for studies involving frequencies of around 1700 Hz or above,
the 16384 Hz data should be used instead".

scipy's default IIR path is cheby1(8, 0.05 dB, Wn=0.8/q), i.e. an order-8
Chebyshev type I with 0.05 dB (+/-0.577%) passband ripple and the cutoff at
only 80% of the output Nyquist.  This script confirms the identification by
reproducing their published file from the 16 kHz source.
"""
import os
import sys

import h5py
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_sampler.filters import design, filter_block  # noqa: E402


def main(f16, f4, guard_s=15):
    x16 = h5py.File(f16)["strain/Strain"][:]
    x4 = h5py.File(f4)["strain/Strain"][:]
    ok = np.isfinite(x4) & np.isfinite(x16).reshape(-1, 4).all(axis=1)
    d = np.diff(np.concatenate(([0], ok.view(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    i = np.argmax(e - s)
    a, b = s[i], e[i]
    seg16 = x16[a * 4:b * 4].astype(float)
    theirs = x4[a:b].astype(float)

    cheby = signal.decimate(seg16, 4, ftype="iir", zero_phase=True)
    taps, _ = design(4096)
    ours = filter_block(seg16, taps)[::4]

    g = slice(guard_s * 4096, -guard_s * 4096)
    den = theirs[g].std()
    print(f"common science segment: {len(theirs)/4096:.0f} s\n")
    print(f"  scipy.signal.decimate(x,4,zero_phase=True) vs GWOSC : "
          f"{np.abs(cheby[g]-theirs[g]).std()/den:.3e}  <- identification")
    print(f"  our Kaiser FIR                             vs GWOSC : "
          f"{np.abs(ours[g]-theirs[g]).std()/den:.3e}")

    bq, aq = signal.cheby1(8, 0.05, 0.8 / 4)
    w, H = signal.freqz(bq, aq, worN=200000, fs=16384)
    A2 = np.abs(H) ** 2          # zero_phase applies the filter twice
    print(f"\n  predicted response of cheby1(8, 0.05 dB, Wn=0.8/4), "
          f"cutoff {0.8/4*8192:.1f} Hz:")
    for f in (1638, 1700, 1800, 2000):
        j = np.argmin(abs(w - f))
        print(f"    {f:5d} Hz  {A2[j]:.4f}  ({20*np.log10(max(A2[j],1e-12)):7.2f} dB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: identify_gwosc_filter.py <16KHZ file> <4KHZ file>")
    main(sys.argv[1], sys.argv[2])
