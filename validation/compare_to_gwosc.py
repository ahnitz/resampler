#!/usr/bin/env python3
"""
Measure the effective anti-alias filter GWOSC used for its official
16384 -> 4096 Hz product, and compare it against the filter in this package.

The trick: both products derive from the same 16 kHz realisation, so the
ratio of their Welch PSDs is a (nearly noise-free) estimate of |H(f)|^2.
"""
import argparse
import os
import subprocess
import sys

import h5py
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_sampler.filters import design, filter_block  # noqa: E402


def load(p):
    with h5py.File(p, "r") as f:
        s = f["strain/Strain"]
        return s[:], int(round(1 / float(s.attrs["Xspacing"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file16", help="a GWOSC *_16KHZ_* strain file")
    ap.add_argument("--file4", help="matching *_4KHZ_* file (downloaded if absent)")
    ap.add_argument("--fmin", type=float, default=20.0,
                    help="low edge of the passband statistic")
    ap.add_argument("--seg", type=float, default=8.0,
                    help="Welch segment length in seconds (longer = finer low-f)")
    ap.add_argument("--freqs", type=str, default="",
                    help="comma-separated frequencies to tabulate")
    a = ap.parse_args()

    f4 = a.file4
    if not f4:
        base = os.path.basename(a.file16).replace("16KHZ", "4KHZ")
        gps = int(base.rsplit("-", 2)[1])
        run = base.split("_")[2]
        root = (f"/gwdata/{run}/{run}_4KHZ_R1/STRAIN_HDF" if run.startswith("O4")
                else f"/gwdata/{run}/strain.4k/hdf.v1")
        det = base.split("-")[1].split("_")[0]
        remote = f"{root}/{det}/{gps // 4194304 * 4194304}/{base}"
        f4 = os.path.join("/tmp", base)
        if not os.path.exists(f4):
            subprocess.run(["pelican", "object", "get", "osdf://" + remote, f4],
                           check=True)

    x16, fs16 = load(a.file16)
    x4, fs4 = load(f4)
    R = fs16 // fs4

    ok = np.isfinite(x4) & np.isfinite(x16).reshape(-1, R).all(axis=1)
    d = np.diff(np.concatenate(([0], ok.view(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    i = np.argmax(e - s)
    lo, hi = s[i], e[i]
    s4 = x4[lo:hi].astype(float)
    s16 = x16[lo * R:hi * R].astype(float)
    print(f"comparing {len(s4)/fs4:.0f} s of common science data\n")

    nper = 1 << int(np.floor(np.log2(fs4 * a.seg)))
    f, p4 = signal.welch(s4, fs4, nperseg=nper)
    f16, p16 = signal.welch(s16, fs16, nperseg=nper * R)
    m = f <= fs4 / 2
    f = f[m]
    gwosc = np.sqrt(p4[m] / np.interp(f, f16, p16))

    taps, pb = design(fs4, fs16)
    mine = filter_block(s16, taps)[::R]
    _, pm = signal.welch(mine, fs4, nperseg=nper)
    ours = np.sqrt(pm[m] / np.interp(f, f16, p16))

    print(f"{'freq Hz':>9} {'GWOSC':>10} {'dB':>8} | {'ours':>10} {'dB':>9}")
    flist = ([float(v) for v in a.freqs.split(",")] if a.freqs else
             [10, 50, 100, 300, 500, 1000, 1500, 1600, 1700, 1800, 1900, 2000])
    for fq in flist:
        j = np.argmin(abs(f - fq))
        g_, o_ = gwosc[j], ours[j]
        print(f"{f[j]:9.1f} {g_:10.5f} {20*np.log10(max(g_,1e-99)):8.2f} | "
              f"{o_:10.5f} {20*np.log10(max(o_,1e-99)):9.2f}")

    band = (f > a.fmin) & (f < pb)
    print(f"\npassband {a.fmin:.0f}-{pb:.0f} Hz, deviation from unity gain:")
    print(f"  GWOSC : {np.abs(gwosc[band]-1).max():.3e}")
    print(f"  ours  : {np.abs(ours[band]-1).max():.3e}")
    stop = f >= fs4 / 2 * 0.999
    if stop.any():
        print(f"at output Nyquist ({fs4/2:.0f} Hz): GWOSC "
              f"{20*np.log10(max(gwosc[stop][0],1e-99)):.1f} dB, "
              f"ours {20*np.log10(max(ours[stop][0],1e-99)):.1f} dB")


if __name__ == "__main__":
    main()
