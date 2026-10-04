# data_sampler

Reduce GWOSC 16384 Hz strain to a lower sample rate — properly.

## Why

GWOSC publishes a 4096 Hz product alongside the 16384 Hz data, but its
anti-alias filter is loosely specified. Measured against the 16 kHz source
(`validation/compare_to_gwosc.py`, O4a H1, 261 s of science data):

| | GWOSC 16k→4k | this package |
|---|---|---|
| passband ripple (20–2000 Hz) | 9.9e-1 | **3.1e-8** |
| response at 1700 Hz | −2.11 dB | **0.00 dB** |
| response at 1800 Hz | −14.62 dB | **0.00 dB** |
| stopband at Nyquist | −47.1 dB | **−117.5 dB** |

Two consequences for the official product: it is only usable to ~1650 Hz
rather than 2048 Hz, and its ~−44 dB stopband lets out-of-band power fold
back at the ~0.6% amplitude level. Even for a 2048 Hz target — where the bad
rolloff lies above the band you keep — about **1% of frequency-dependent
amplitude error** (−39 dB) is inherited across 10–700 Hz. Starting from the
16 kHz files avoids all of it.

## Install

```bash
pip install -e .
```

Requires the [`pelican`](https://pelicanplatform.org) client on `PATH` for
downloads.

## Use

```bash
# one GWOSC file's worth, at 2048 Hz
data-sampler --run O4a --detector H1 --start 1368354816 --rate 2048

# an arbitrary span — not tied to the 4096 s file layout
data-sampler --run O4a --detector H1 --start 1368353000 --end 1368357000 \
             --rate 2048 -o out.hdf5

# reuse 16 kHz files you already have instead of downloading
data-sampler --run O4a --detector H1 --start 1368354816 --rate 4096 \
             --local-dir /mnt/drive/gwosc
```

`--rate` may be any integer divisor of 16384.

```python
from data_sampler import load_span, reduce_strain, write_reduced
x, dq, inj, template, sources = load_span("O4a", "H1", t0, t1, cache="/tmp/c")
y, edge, taps, pb = reduce_strain(x, 2048)
```

## Output format

The GWOSC HDF5 layout is preserved verbatim — `meta`, `quality/simple`,
`quality/injections`, `strain` — so existing readers work unchanged. Strain is
written as **float32**; quality masks are sliced to the requested span and
`Duration` / `GPSstart` / `UTCstart` are updated.

Added, purely additively:

| path | meaning |
|---|---|
| `quality/edge/EdgeLevel` | per-sample filter provenance, see below |
| `provenance/SourceFiles` | the 16 kHz files this was built from |
| `provenance/FilterTaps` | the exact FIR used |

## Edge handling

Filtering follows **contiguous science data, never file boundaries**. To
produce a span the reader asks for, real data from the neighbouring files is
pulled in as filter context, so an internal file seam is invisible and nothing
is discarded there. A continuous 137 Hz tone spliced across a file boundary
reconstructs to 6e-10 of its amplitude with zero samples flagged.

At a **true science edge** no real data exists beyond it, so the input is
odd-reflected (not zero-padded, which would inject a step discontinuity and
ring far worse). Nothing is trimmed — the full segment is emitted, plus
`--extend` seconds of filter ring-out past the edge — and every sample carries
a level saying how it was produced:

| `EdgeLevel` | meaning | measured accuracy |
|---|---|---|
| 0 | clean — filter saw only real data | 6e-10 |
| 1 | real data, filter leaned on reflected padding | 7e-5 |
| 2 | beyond real data; filter ring-out only | not faithful, use knowingly |

Level 1 and 2 each span only ~0.5 s per science edge.

## float32 and dynamic range

float32 storage is comfortably sufficient and **no dynamic-range factor is
needed for precision**. Measured on real O4a strain, float32 round-trip
quantisation noise sits **76 dB below the data** across 20–1024 Hz
(amplitude ratio 1.6e-4). Scaling by a power of two changes this by exactly
nothing — floating point stores a mantissa, not an absolute step.

A scale factor matters only for *downstream arithmetic*: strain² ≈ 2.3e-36 is
only ~200× above the float32 minimum normal (1.18e-38), so code that squares
strain in float32 will underflow. `provenance/DynRangeFac` records the applied
factor (1.0 — data is stored unscaled, GWOSC-compatible); apply something like
2**69 in your own pipeline if you compute in float32.

## Storage

16 kHz GWOSC strain is float64: 131 kB/s. At 2048 Hz float32 it is 8 kB/s —
**16× smaller**. The complete 16 kHz catalogue (O1–O4b, 20.6 TB) reduces to
roughly **1.3 TB**.

## Tests

```bash
pytest
python validation/compare_to_gwosc.py <a 16KHZ file>
```
