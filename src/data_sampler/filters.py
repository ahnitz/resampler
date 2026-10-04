"""Anti-alias filter design and decimation of gapped strain data."""
from __future__ import annotations

import numpy as np
from scipy import signal

FS_IN = 16384

#: Quality levels recorded alongside the reduced strain.
EDGE_CLEAN, EDGE_PAD, EDGE_EXTRAP = 0, 1, 2
EDGE_DESCRIPTIONS = (
    "clean: filter saw only real data",
    "pad-affected: real data, but the filter leaned on reflected padding",
    "extrapolated: beyond real data; filter ring-out only",
)


def design(fs_out, fs_in=FS_IN, atten_db=180.0, passband_frac=1000.0 / 1024.0):
    """Kaiser-window FIR for decimation to ``fs_out``.

    Returns ``(taps, passband_hz)``. The default spec gives <2e-8 passband
    ripple and better than -150 dB stopband -- compare GWOSC's own 16k->4k
    filter at 1.1e-2 ripple and -44 dB.
    """
    if fs_in % fs_out:
        raise ValueError(f"{fs_in} Hz is not an integer multiple of {fs_out} Hz")
    nyq = fs_out / 2.0
    pb = passband_frac * nyq
    ntaps, beta = signal.kaiserord(atten_db, (nyq - pb) / (fs_in / 2.0))
    ntaps = int(ntaps) | 1  # odd -> exact linear phase
    taps = signal.firwin(ntaps, 0.5 * (pb + nyq), fs=fs_in, window=("kaiser", beta))
    return taps, pb


def filter_block(x, taps):
    """Zero-phase FIR over one contiguous block, odd-reflected at both ends.

    Odd reflection continues the signal's local trend, so it does not inject
    the step discontinuity that zero-padding would.
    """
    pad = len(taps) // 2
    k = min(pad, len(x) - 1)
    left = np.concatenate((np.full(pad - k, x[0]), x[1:k + 1][::-1]))
    right = np.concatenate((x[-k - 1:-1][::-1], np.full(pad - k, x[-1])))
    xp = np.concatenate((2 * x[0] - left, x, 2 * x[-1] - right))
    return signal.fftconvolve(xp, taps, mode="same")[pad:pad + len(x)]


def contiguous_runs(ok):
    """[(start, stop), ...] for each maximal True run in boolean ``ok``."""
    d = np.diff(np.concatenate(([0], ok.view(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def reduce_strain(x, fs_out, fs_in=FS_IN, extend_s=0.25, taps=None, **kw):
    """Reduce 16 kHz ``x`` (NaN outside science data) to ``fs_out``.

    Filtering follows contiguous science data, never file boundaries. Nothing
    is trimmed: the full span of every segment is emitted, plus ``extend_s``
    of filter ring-out past each true edge, with an ``edge`` level marking how
    each output sample was produced.

    Returns ``(y, edge, taps, passband_hz)``.
    """
    M = fs_in // fs_out
    if taps is None:
        taps, pb = design(fs_out, fs_in, **kw)
    else:
        pb = kw.get("passband_hz", float("nan"))
    pad, ext = len(taps) // 2, int(extend_s * fs_in)

    n_out = len(x) // M
    y = np.full(n_out, np.nan)
    edge = np.zeros(n_out, np.uint8)

    def mark(lo, hi, val):
        lo, hi = max(lo, 0) // M, min(hi, len(x)) // M
        if hi > lo:
            edge[lo:hi] = np.maximum(edge[lo:hi], val)

    for a, b in contiguous_runs(np.isfinite(x)):
        if b - a < 2 * M:
            continue
        seg = x[a:b].astype(np.float64)
        core = filter_block(seg, taps)
        ea, eb = max(a - ext, 0), min(b + ext, len(x))
        buf = np.empty(eb - ea)
        buf[a - ea:a - ea + len(core)] = core
        if a > ea:
            buf[:a - ea] = filter_block(
                np.concatenate((np.full(a - ea, seg[0]), seg[:pad])), taps)[:a - ea]
        if eb > b:
            buf[len(buf) - (eb - b):] = filter_block(
                np.concatenate((seg[-pad:], np.full(eb - b, seg[-1]))), taps)[-(eb - b):]
        o_lo, o_hi = ea // M, eb // M
        off = o_lo * M - ea
        y[o_lo:o_hi] = buf[off: off + (o_hi - o_lo) * M: M]
        mark(a, a + pad, EDGE_PAD)
        mark(b - pad, b, EDGE_PAD)
        mark(ea, a, EDGE_EXTRAP)
        mark(b, eb, EDGE_EXTRAP)
    return y, edge, taps, pb
