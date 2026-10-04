"""Reduce GWOSC 16 kHz strain to a lower sample rate, properly.

The official GWOSC 4096 Hz product is produced with a loose anti-alias
filter (~1.1e-2 passband ripple, only -44 dB at Nyquist).  This package
re-derives reduced-rate data straight from the 16384 Hz files using a
filter specified to <1e-8 ripple and better than -150 dB stopband, while
keeping the GWOSC file layout so existing readers work unchanged.
"""
from .filters import (EDGE_CLEAN, EDGE_EXTRAP, EDGE_PAD, EDGE_DESCRIPTIONS,
                      design, reduce_strain)
from .gwosc_io import read_file, write_reduced
from .osdf import index, load_span

__version__ = "0.1.0"
__all__ = ["design", "reduce_strain", "read_file", "write_reduced",
           "index", "load_span", "EDGE_CLEAN", "EDGE_PAD", "EDGE_EXTRAP",
           "EDGE_DESCRIPTIONS", "__version__"]
