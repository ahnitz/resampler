"""Reading and writing GWOSC-format strain HDF5 files.

The reduced files this module writes keep the GWOSC layout verbatim --
``meta``, ``quality/simple``, ``quality/injections`` and ``strain`` -- so
existing readers work unchanged.  Added, and purely additive:

``quality/edge/EdgeLevel``  per-sample filter provenance (see filters.py)
``provenance/``             source files, filter spec and taps
"""
from __future__ import annotations

import datetime as _dt

import h5py
import numpy as np

FS_IN = 16384
FILE_DUR = 4096

# GPS-UTC offset; (gps_at_which_it_took_effect, leap_seconds)
_LEAPS = ((0, 0), (46828800, 1), (78364801, 2), (109900802, 3), (173059203, 4),
          (252028804, 5), (315187205, 6), (346723206, 7), (393984007, 8),
          (425520008, 9), (457056009, 10), (504489610, 11), (551750411, 12),
          (599184012, 13), (820108813, 14), (914803214, 15), (1025136015, 16),
          (1119744016, 17), (1167264017, 18))


def gps_to_utc(gps: int) -> str:
    leap = max(l for g, l in _LEAPS if gps >= g)
    t = _dt.datetime(1980, 1, 6, tzinfo=_dt.timezone.utc) + \
        _dt.timedelta(seconds=int(gps) - leap)
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def read_file(path):
    """Return a dict with strain, 1 Hz quality masks and all metadata."""
    with h5py.File(path, "r") as f:
        st = f["strain/Strain"]
        out = {
            "strain": st[:],
            "strain_attrs": dict(st.attrs),
            "gps": int(st.attrs["Xstart"]),
            "fs": int(round(1.0 / float(st.attrs["Xspacing"]))),
            "meta": {k: f["meta"][k][()] for k in f["meta"]},
            "dq": f["quality/simple/DQmask"][:],
            "inj": f["quality/injections/Injmask"][:],
            "dq_desc": f["quality/simple/DQDescriptions"][:],
            "dq_short": f["quality/simple/DQShortnames"][:],
            "inj_desc": f["quality/injections/InjDescriptions"][:],
            "inj_short": f["quality/injections/InjShortnames"][:],
        }
        for k, p in (("dq_gwoscmeta", "quality/simple/GWOSCmeta"),
                     ("inj_gwoscmeta", "quality/injections/GWOSCmeta"),
                     ("strain_gwoscmeta", "strain/GWOSCmeta")):
            if p in f:
                out[k] = f[p][()]
    return out


def write_reduced(path, *, y, edge, gps_start, fs_out, duration, template,
                  dq, inj, taps, passband_hz, sources, compression="gzip",
                  complevel=4):
    """Write a reduced file in the GWOSC layout.

    ``template`` is a dict from :func:`read_file` supplying the metadata to
    carry over; ``sources`` is the list of input filenames used.
    """
    with h5py.File(path, "w") as g:
        st = g.create_dataset("strain/Strain", data=np.asarray(y, np.float32),
                              compression=compression,
                              compression_opts=complevel, shuffle=True)
        st.attrs.update({
            "Npoints": len(y), "Xlabel": "GPS time",
            "Xspacing": 1.0 / fs_out, "Xstart": int(gps_start),
            "Xunits": "second", "Ylabel": "Strain", "Yunits": "",
        })
        if "strain_gwoscmeta" in template:
            g.create_dataset("strain/GWOSCmeta", data=template["strain_gwoscmeta"])

        m = template["meta"]
        mg = g.create_group("meta")
        for k, v in m.items():
            if k == "Duration":
                v = np.int64(duration)
            elif k == "GPSstart":
                v = np.int64(gps_start)
            elif k == "UTCstart":
                v = gps_to_utc(gps_start)
            mg.create_dataset(k, data=v)

        q = g.create_group("quality/simple")
        q.create_dataset("DQmask", data=dq, compression=compression)
        q.create_dataset("DQDescriptions", data=template["dq_desc"])
        q.create_dataset("DQShortnames", data=template["dq_short"])
        if "dq_gwoscmeta" in template:
            q.create_dataset("GWOSCmeta", data=template["dq_gwoscmeta"])

        j = g.create_group("quality/injections")
        j.create_dataset("Injmask", data=inj, compression=compression)
        j.create_dataset("InjDescriptions", data=template["inj_desc"])
        j.create_dataset("InjShortnames", data=template["inj_short"])
        if "inj_gwoscmeta" in template:
            j.create_dataset("GWOSCmeta", data=template["inj_gwoscmeta"])

        # ---- additive extensions ----
        from .filters import EDGE_DESCRIPTIONS
        e = g.create_dataset("quality/edge/EdgeLevel", data=np.asarray(edge, np.uint8),
                             compression=compression, shuffle=True)
        e.attrs["Descriptions"] = np.array(EDGE_DESCRIPTIONS,
                                           dtype=h5py.string_dtype())
        e.attrs["Xspacing"] = 1.0 / fs_out
        e.attrs["Xstart"] = int(gps_start)

        p = g.create_group("provenance")
        p.create_dataset("SourceFiles",
                         data=np.array(list(sources), dtype=h5py.string_dtype()))
        p.create_dataset("FilterTaps", data=np.asarray(taps, np.float64),
                         compression=compression)
        p.attrs.update({
            "OriginalSampleRate": FS_IN, "SampleRate": int(fs_out),
            "FilterNtaps": len(taps), "FilterPassbandHz": float(passband_hz),
            "FilterType": "Kaiser-window linear-phase FIR, zero phase",
            "EdgePadding": "odd reflection",
            "ConvolutionMethod": "scipy.signal.oaconvolve",
            "DynRangeFac": 1.0,
            "Producer": "data_sampler",
        })
