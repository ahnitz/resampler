#!/usr/bin/env python3
"""
Bulk: download every GWOSC 16 kHz file in priority order, write the reduced
version, and delete the 16 kHz original.

Each (run, detector) is an independent sequential stream so that neighbouring
files -- needed as filter context -- stay in that stream's cache.  Streams run
in parallel, which is what keeps the network busy.
"""
import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_sampler.filters import FS_IN, design, reduce_strain
from data_sampler.gwosc_io import FILE_DUR, write_reduced
from data_sampler.osdf import chunks, index, load_span

PRIORITY = ["O4a", "O4b", "O3a", "O3b", "O2", "O1"]
DETECTORS = ["H1", "L1", "V1"]

lock = threading.Lock()
stats = {"files": 0, "bytes_out": 0, "skipped": 0, "failed": 0}
t0 = time.time()


def log(msg):
    with lock:
        print(f"[{time.time()-t0:8.0f}s] {msg}", flush=True)


def out_path(dest, run, det, chunk, gps, rate):
    return os.path.join(dest, run, det, str(chunk),
                        f"{det[0]}-{det}_DATASAMPLER_{run}_{rate}HZ-{gps}-{FILE_DUR}.hdf5")


def valid(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        return False
    try:
        with h5py.File(path, "r") as f:
            return "strain/Strain" in f
    except Exception:
        return False


def run_stream(run, det, args):
    cache = os.path.join(args.cache, f"{run}_{det}")
    taps, pb = design(args.rate, atten_db=args.atten_db)
    ctx = int(math.ceil((len(taps) // 2) / FS_IN)) + 2

    try:
        cs = chunks(run, det)
    except Exception as e:
        log(f"{run} {det}: cannot list chunks ({e})")
        return
    if not cs:
        log(f"{run} {det}: no data")
        return

    # full file list for this stream
    files = {}
    for c in cs:
        files.update(index(run, det, c, c + 4194304 + FILE_DUR))
    gpss = sorted(files)
    log(f"{run} {det}: {len(gpss)} files")

    if args.max_files:
        gpss = gpss[:args.max_files]
    for k, gps in enumerate(gpss):
        chunk = [c for c in cs if c <= gps][-1]
        dst = out_path(args.dest, run, det, chunk, gps, args.rate)
        if valid(dst):
            with lock:
                stats["skipped"] += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            x, dq, inj, template, sources = load_span(
                run, det, gps - ctx, gps + FILE_DUR + ctx, cache)
            y, edge, taps_, pb_ = reduce_strain(
                x, args.rate, taps=taps, passband_hz=pb)
            i0 = ctx * args.rate
            tmp = dst + ".part"
            write_reduced(tmp, y=y[i0:i0 + FILE_DUR * args.rate],
                          edge=edge[i0:i0 + FILE_DUR * args.rate],
                          gps_start=gps, fs_out=args.rate, duration=FILE_DUR,
                          template=template, dq=dq[ctx:ctx + FILE_DUR],
                          inj=inj[ctx:ctx + FILE_DUR], taps=taps_,
                          passband_hz=pb_, sources=sources)
            os.replace(tmp, dst)
            with lock:
                stats["files"] += 1
                stats["bytes_out"] += os.path.getsize(dst)
                n = stats["files"]
            if n % 20 == 0:
                log(f"{n} written, {stats['bytes_out']/1e9:.1f} GB out, "
                    f"{stats['skipped']} skipped, {stats['failed']} failed")
        except Exception as e:
            with lock:
                stats["failed"] += 1
            log(f"FAIL {run} {det} {gps}: {type(e).__name__}: {str(e)[:200]}")

        # prune cache: keep only files still needed as context
        if not args.keep_16k:
            keep = {os.path.basename(files[g])
                    for g in gpss[max(k, 0):k + 3] if g in files}
            for fn in os.listdir(cache) if os.path.isdir(cache) else []:
                if fn not in keep and not fn.endswith(".part"):
                    try:
                        os.remove(os.path.join(cache, fn))
                    except OSError:
                        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="output root")
    ap.add_argument("--cache", default="/tmp/ds_bulk", help="16 kHz scratch (fast disk)")
    ap.add_argument("--rate", type=int, default=2048)
    ap.add_argument("--atten-db", type=float, default=180.0)
    ap.add_argument("--runs", default=",".join(PRIORITY))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep-16k", action="store_true")
    ap.add_argument("--max-files", type=int, default=0,
                    help="stop each stream after N files (smoke test)")
    args = ap.parse_args()

    streams = [(r, d) for r in args.runs.split(",") for d in DETECTORS]
    log(f"{len(streams)} candidate streams, {args.workers} workers, "
        f"-> {args.dest} at {args.rate} Hz")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda rd: run_stream(rd[0], rd[1], args), streams))
    log(f"DONE files={stats['files']} skipped={stats['skipped']} "
        f"failed={stats['failed']} out={stats['bytes_out']/1e12:.3f} TB")


if __name__ == "__main__":
    main()
