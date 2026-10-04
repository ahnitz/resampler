# data_sampler

Reduce GWOSC 16384 Hz strain to a lower sample rate — properly.

## Why

GWOSC publishes a 4096 Hz product alongside the 16384 Hz data, but its
anti-alias filter is loosely specified. Measured against the 16 kHz source
(`validation/compare_to_gwosc.py`, O4a H1, 261 s of science data):

| | GWOSC 16k→4k | this package |
|---|---|---|
| response at 5–30 Hz | −0.10 dB (0.9886) | **0.00 dB (1.0000)** |
| response at 1700 Hz | −2.11 dB | **0.00 dB** |
| response at 1800 Hz | −14.62 dB | **0.00 dB** |
| deviation from unity, 5–2000 Hz | 9.9e-1 | **3.8e-9** |
| stopband at Nyquist | −45.6 dB | **−117.3 dB** |

Their filter is equiripple at roughly ±0.57% about unity, and that ripple runs
all the way down to DC — the response is a flat 0.98857 from 5 Hz through
30 Hz. So the ~1% amplitude error is not confined to the top of the band; it
is present at the low frequencies where most searches carry their SNR.

Two consequences for the official product: it is only usable to ~1650 Hz
rather than 2048 Hz, and its ~−44 dB stopband lets out-of-band power fold
back at the ~0.6% amplitude level. Even for a 2048 Hz target — where the bad
rolloff lies above the band you keep — about **1% of frequency-dependent
amplitude error** (−39 dB) is inherited across 10–700 Hz. Starting from the
16 kHz files avoids all of it.

## Does it matter in practice?

`validation/injection_recovery.py` injects an inspiral into real O4a strain
and matched-filters the result through three pipelines: the 16 kHz data
(truth), this package, and GWOSC's own 4096 Hz product.

| band | data_sampler | GWOSC 4096 Hz |
|---|---|---|
| 20–922 Hz | −0.00 % | −0.12 % |
| 1400–1843 Hz | −0.18 % | **−28.2 %** |
| 1600–1843 Hz | **−0.00 %** | **−25.9 %** |

Two honest qualifications:

* **Below ~1 kHz their product is fine for matched filtering.** Matched-filter
  SNR is invariant to an overall gain -- the data and its estimated PSD scale
  together -- so GWOSC's flat 1.14% deficit cancels, and only their ripple
  costs anything, about 0.1%. Their 1% error matters for calibration-sensitive
  work, not for SNR.
* **Their boundary handling has no pathology.** `validation/boundary_test.py`
  shows they lose no science time, and their deviation is flat with distance
  from a science edge (7e-3 at 0–0.05 s, 1.2e-2 at 20–60 s) rather than
  growing towards it.

The real cost of their filter is **bandwidth**: a quarter of the SNR is gone
above 1600 Hz, which is where post-merger and neutron-star physics lives.

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
odd-reflected. What actually matters is *not zero-padding*:
`validation/padding_bench.py` measures each option against a full-context
reference, and zero-padding is ~40x worse than anything else because it
asserts an instantaneous step to zero that the filter then rings on. Edge-hold
(1.05e-3), odd reflection (1.28e-3) and mirror (1.88e-3) are all within a
factor of two, and all are exact beyond ntaps//2 samples (0.25 s) from the
edge. Nothing is trimmed — the full segment is emitted, plus
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
strain in float32 will underflow. That is a property of strain's physical
scale, not of this format — the 16 kHz float64 originals underflow the moment
you cast them.

Analysis codes already apply their own `DYN_RANGE_FAC`, so these files are
stored **unscaled** and `provenance/DynRangeFac` is `1.0`. Keep it that way:
pre-scaling the files would double-apply the factor and would break
byte-compatibility with GWOSC readers. The attribute exists so a reader can
confirm nothing was applied.

## Storage

16 kHz GWOSC strain is float64: 131 kB/s. At 2048 Hz float32 it is 8 kB/s —
**16× smaller**. The complete 16 kHz catalogue (O1–O4b, 20.6 TB) reduces to
roughly **1.3 TB**.

## Compression

Strain is stored gzip level 4 with the shuffle filter, matching GWOSC.
`validation/compression_bench.py` shows this is at the practical optimum:
shuffle is the entire win (-16%), the gzip level is irrelevant (level 9 saves
0.1% for twice the write cost), and zstd/blosc gain nothing while requiring
`hdf5plugin` just to read the files.

Real savings exist only by going lossy -- truncating the float32 mantissa to
16 bits saves 24% with added noise still 104 dB below the data. That is
deliberately not done: space is not the binding constraint, and compression
has no effect on the download time that actually dominates.

## Tests

```bash
pytest
python validation/compare_to_gwosc.py <a 16KHZ file>
```
