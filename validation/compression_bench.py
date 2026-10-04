#!/usr/bin/env python3
"""Benchmark HDF5 codecs for the reduced strain. Conclusion: gzip+shuffle is
at the practical optimum; shuffle is the whole win and the gzip level is
irrelevant. zstd/blosc gain nothing and would need hdf5plugin to READ."""
import sys, os, time, numpy as np, h5py
sys.path.insert(0,'src')
from data_sampler.filters import reduce_strain
B="/run/media/ahnitz/Seagate Backup Plus Drive/gwosc/O4a/H1/1367343104/"
CASES=[("none",{}),("lzf",{"compression":"lzf"}),("lzf+shuf",{"compression":"lzf","shuffle":True})]
for lvl in (1,4,6,9):
    CASES.append((f"gzip{lvl}",{"compression":"gzip","compression_opts":lvl}))
    CASES.append((f"gzip{lvl}+shuf",{"compression":"gzip","compression_opts":lvl,"shuffle":True}))
for fn,label in [("H-H1_GWOSC_O4a_16KHZ_R1-1368350720-4096.hdf5","100% science"),
                 ("H-H1_GWOSC_O4a_16KHZ_R1-1368268800-4096.hdf5","6% science")]:
    x=h5py.File(B+fn)['strain/Strain'][:]
    y,edge,_,_=reduce_strain(x,2048); y32=y.astype(np.float32)
    print(f"\n=== {label} ({np.isfinite(y).mean()*100:.1f}% finite, {len(y32)} samples) ===")
    print(f"{'codec':16} {'size':>10} {'vs gzip4+shuf':>14} {'write s':>9} {'read s':>8}")
    base=None
    for name,kw in CASES:
        p=f"/tmp/cb_{name}.h5"
        t=time.time()
        with h5py.File(p,'w') as f: f.create_dataset("s",data=y32,**kw)
        tw=time.time()-t
        t=time.time()
        with h5py.File(p,'r') as f: _=f["s"][:]
        tr=time.time()-t
        sz=os.path.getsize(p)
        if name=="gzip4+shuf": base=sz
        print(f"{name:16} {sz/1e6:8.2f} MB {'':>14} {tw:9.2f} {tr:8.2f}" if base is None
              else f"{name:16} {sz/1e6:8.2f} MB {100*(sz/base-1):+13.1f}% {tw:9.2f} {tr:8.2f}")
        os.remove(p)
