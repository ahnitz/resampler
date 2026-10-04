"""Locating and fetching GWOSC 16 kHz strain from the OSDF via pelican."""
from __future__ import annotations

import os
import re
import subprocess
import time

import numpy as np

from .gwosc_io import FILE_DUR, FS_IN, read_file

CHUNK = 4194304  # GPS span of one OSDF directory

ROOTS = {
    "O4a": "/gwdata/O4a/O4a_16KHZ_R1/STRAIN_HDF",
    "O4b": "/gwdata/O4b/O4b_16KHZ_R1/STRAIN_HDF",
    "O3a": "/gwdata/O3a/strain.16k/hdf.v1",
    "O3b": "/gwdata/O3b/strain.16k/hdf.v1",
    "O2": "/gwdata/O2/strain.16k/hdf.v1",
    "O1": "/gwdata/O1/strain.16k/hdf.v1",
}

_NAME = re.compile(r"-(\d+)-(\d+)\.hdf5$")
_LS_CACHE: dict = {}


def _ls(path):
    r = subprocess.run(["pelican", "object", "ls", f"osdf://{path}/"],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return []
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def chunks(run, det):
    """Sorted GPS-named subdirectories for a detector.

    These are *not* reliably aligned to multiples of ``CHUNK`` -- O4 uses
    exact multiples, O3 does not -- so they are discovered, never computed.
    """
    key = ("chunks", run, det)
    if key not in _LS_CACHE:
        _LS_CACHE[key] = sorted(
            int(c) for c in _ls(f"{ROOTS[run]}/{det}") if c.isdigit())
    return _LS_CACHE[key]


def index(run, det, gps_lo, gps_hi):
    """``{gps_start: osdf_path}`` for files overlapping ``[gps_lo, gps_hi)``."""
    root = ROOTS[run]
    found = {}
    all_c = chunks(run, det)
    # every chunk that could hold an overlapping file: those starting before
    # gps_hi, plus the one covering gps_lo
    cand = [c for c in all_c if c < gps_hi]
    cand = cand[-1:] if not cand else cand
    cand = [c for c in cand if c >= (max([x for x in all_c if x <= gps_lo],
                                         default=cand[0]))]
    for chunk in cand:
        key = (run, det, chunk)
        if key not in _LS_CACHE:
            _LS_CACHE[key] = _ls(f"{root}/{det}/{chunk}")
        for name in _LS_CACHE[key]:
            m = _NAME.search(name)
            if m:
                g, d = int(m.group(1)), int(m.group(2))
                if g + d > gps_lo and g < gps_hi:
                    found[g] = f"{root}/{det}/{chunk}/{name}"
    return found


def fetch(remote, cache, attempts=6, base_delay=5.0):
    """Download ``remote`` into ``cache``, returning the local path.

    OSDF caches intermittently answer "temporarily unavailable", so transient
    failures are retried with exponential backoff -- a multi-day bulk run will
    hit plenty of them.
    """
    local = os.path.join(cache, os.path.basename(remote))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    os.makedirs(cache, exist_ok=True)
    tmp = local + ".part"
    last = ""
    for i in range(attempts):
        if i:
            time.sleep(min(base_delay * 2 ** (i - 1), 300.0))
        try:
            r = subprocess.run(["pelican", "object", "get",
                                f"osdf://{remote}", tmp],
                               capture_output=True, text=True, timeout=7200)
        except subprocess.TimeoutExpired:
            last = "timeout"
            continue
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, local)
            return local
        last = (r.stderr or r.stdout)[-300:]
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    raise RuntimeError(f"download failed after {attempts} attempts: "
                       f"{remote}\n{last}")


def scan_local(dirs):
    """``{gps_start: path}`` for 16 kHz files already on disk."""
    have = {}
    for d in dirs or ():
        for dirpath, _, names in os.walk(d):
            for n in names:
                m = _NAME.search(n)
                if m:
                    have[int(m.group(1))] = os.path.join(dirpath, n)
    return have


def load_span(run, det, gps_lo, gps_hi, cache, local_dirs=(), verbose=False):
    """Assemble 16 kHz strain and 1 Hz quality masks over ``[gps_lo, gps_hi)``.

    Missing coverage is NaN in the strain and zero in the masks. Returns
    ``(x, dq, inj, template, sources)``.
    """
    have = scan_local(local_dirs)
    idx = index(run, det, gps_lo, gps_hi)
    if not idx:
        raise RuntimeError(f"no {run} {det} files found covering "
                           f"[{gps_lo}, {gps_hi})")

    n = (gps_hi - gps_lo) * FS_IN
    x = np.full(n, np.nan)
    dq = np.zeros(gps_hi - gps_lo, np.uint32)
    inj = np.zeros(gps_hi - gps_lo, np.uint32)
    template, sources = None, []

    for g in sorted(idx):
        path = have.get(g) or fetch(idx[g], cache)
        if verbose:
            print(f"  [{'local' if g in have else 'osdf '}] "
                  f"{os.path.basename(path)}", flush=True)
        d = read_file(path)
        if d["fs"] != FS_IN:
            raise RuntimeError(f"{path}: expected {FS_IN} Hz, got {d['fs']}")
        template = template or d
        sources.append(os.path.basename(path))

        s_lo, s_hi = max(gps_lo - g, 0), min(gps_hi - g, FILE_DUR)
        d_lo = max(g - gps_lo, 0)
        x[d_lo * FS_IN: (d_lo + s_hi - s_lo) * FS_IN] = \
            d["strain"][s_lo * FS_IN: s_hi * FS_IN]
        dq[d_lo: d_lo + s_hi - s_lo] = d["dq"][s_lo:s_hi]
        inj[d_lo: d_lo + s_hi - s_lo] = d["inj"][s_lo:s_hi]

    return x, dq, inj, template, sources
