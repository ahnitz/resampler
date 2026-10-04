"""Command line entry point: fetch GWOSC 16 kHz strain and reduce it."""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from .filters import FS_IN, design, reduce_strain
from .gwosc_io import FILE_DUR, write_reduced
from .osdf import ROOTS, load_span


def build_parser():
    p = argparse.ArgumentParser(
        prog="data-sampler",
        description="Download GWOSC 16 kHz strain and write a reduced-rate "
                    "file in the same GWOSC HDF5 format.")
    p.add_argument("--run", required=True, choices=sorted(ROOTS),
                   help="observing run")
    p.add_argument("--detector", required=True, help="e.g. H1, L1, V1")
    p.add_argument("--start", type=int, required=True, help="GPS start second")
    p.add_argument("--end", type=int, help="GPS end second (exclusive)")
    p.add_argument("--duration", type=int,
                   help="length in seconds; alternative to --end")
    p.add_argument("--rate", type=int, default=2048,
                   help="output sample rate, must divide 16384 (default 2048)")
    p.add_argument("--out", "-o", help="output file (default: auto-named)")
    p.add_argument("--cache", default=os.environ.get(
        "DATA_SAMPLER_CACHE", os.path.expanduser("~/.cache/data_sampler")),
        help="where downloaded 16 kHz files are kept")
    p.add_argument("--local-dir", action="append", default=[],
                   help="directory of already-downloaded 16 kHz files "
                        "(repeatable)")
    p.add_argument("--atten-db", type=float, default=180.0,
                   help="stopband attenuation (default 180)")
    p.add_argument("--passband-frac", type=float, default=1000.0 / 1024.0,
                   help="flat fraction of the output Nyquist (default 0.977)")
    p.add_argument("--extend", type=float, default=0.25,
                   help="seconds of flagged filter ring-out emitted past each "
                        "science edge (default 0.25)")
    p.add_argument("--keep-cache", action="store_true",
                   help="keep the downloaded 16 kHz files (default: keep)")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.end is None and a.duration is None:
        a.end = (a.start // FILE_DUR) * FILE_DUR + FILE_DUR
    elif a.end is None:
        a.end = a.start + a.duration
    if a.end <= a.start:
        sys.exit("--end must be after --start")
    if FS_IN % a.rate:
        sys.exit(f"--rate {a.rate} does not divide {FS_IN}")
    say = (lambda *x: None) if a.quiet else (
        lambda *x: print(*x, file=sys.stderr, flush=True))

    taps, pb = design(a.rate, atten_db=a.atten_db,
                      passband_frac=a.passband_frac)
    # whole seconds of real neighbouring data the filter needs on each side
    ctx = int(math.ceil((len(taps) // 2) / FS_IN)) + int(math.ceil(a.extend)) + 1
    lo, hi = a.start - ctx, a.end + ctx

    say(f"{a.run} {a.detector}  [{a.start}, {a.end})  -> {a.rate} Hz")
    say(f"filter: {len(taps)} taps, flat to {pb:.1f} Hz, "
        f"{a.atten_db:.0f} dB stopband; context {ctx} s each side")

    x, dq, inj, template, sources = load_span(
        a.run, a.detector, lo, hi, a.cache, a.local_dir, verbose=not a.quiet)

    y, edge, taps, pb = reduce_strain(
        x, a.rate, extend_s=a.extend, taps=taps, passband_hz=pb)

    # slice back to exactly the requested span
    i0 = (a.start - lo) * a.rate
    i1 = (a.end - lo) * a.rate
    y, edge = y[i0:i1], edge[i0:i1]
    dq = dq[a.start - lo: a.end - lo]
    inj = inj[a.start - lo: a.end - lo]

    out = a.out or (f"{a.detector[0]}-{a.detector}_DATASAMPLER_{a.run}_"
                    f"{a.rate}HZ-{a.start}-{a.end - a.start}.hdf5")
    write_reduced(out, y=y, edge=edge, gps_start=a.start, fs_out=a.rate,
                  duration=a.end - a.start, template=template, dq=dq, inj=inj,
                  taps=taps, passband_hz=pb, sources=sources)

    good = int(np.isfinite(y).sum())
    say(f"wrote {out}")
    say(f"  {good}/{len(y)} finite samples ({100*good/max(len(y),1):.2f}%)  "
        f"clean={int((edge==0).sum())} pad={int((edge==1).sum())} "
        f"extrap={int((edge==2).sum())}")
    say(f"  {os.path.getsize(out)/1e6:.1f} MB from {len(sources)} source file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
