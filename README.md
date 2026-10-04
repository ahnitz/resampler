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

### What GWOSC actually does

They document it, and it checks out exactly. Their page says the downsampling
uses `scipy.signal.decimate`, and scipy's default IIR path is
`cheby1(8, 0.05 dB, Wn=0.8/q)` -- an order-8 Chebyshev type I with the cutoff
at just **80% of the output Nyquist** (1638 Hz) and **0.05 dB passband
ripple**, which is +/-0.577%. That predicts -2.11 dB at 1700 Hz, -14.62 dB at
1800 Hz and -44.07 dB at 2000 Hz; measured on real data the figures are
-2.11, -14.62 and -43.89 dB. `validation/identify_gwosc_filter.py` reproduces
their published file from the 16 kHz source to 2e-12.

So this is not a hidden defect -- it is scipy's default, and GWOSC says
plainly that "for studies involving frequencies of around 1700 Hz or above,
the 16384 Hz data should be used instead". This package is a way to take that
advice without carrying 16 kHz volumes.

| | GWOSC | data_sampler |
|---|---|---|
| filter | order-8 Chebyshev I (IIR) | Kaiser-window FIR, ~8183 taps |
| passband ripple | 0.05 dB (+/-0.577%) | <2e-9 |
| cutoff | 80% of output Nyquist | 97.7% |
| stopband at Nyquist | -44 dB | -179 dB |
| phase | zero-phase (filtfilt) | zero-phase (symmetric FIR) |
| at segment edges | 8 s real padding "not always available" | real neighbour-file context; odd reflection only at true edges, flagged per sample |

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

### Why not shorten the kernel near the edge instead?

Tempting, and more principled-sounding: an output sample `d` samples from an
edge can only draw on `d` real samples, so use the longest filter that *fits*
and invent nothing. `design_ladder` builds exactly that, and
`reduce_strain_tapered` applies it.

Measured, it loses. Restricted to 20-900 Hz -- strictly inside every ladder
passband, so the short filters are not penalised for the band they openly
drop -- reflection is better at every distance:

| window from edge | reflect | taper |
|---|---|---|
| 37-61 ms | **2.7e-06** | 1.6e-05 |
| 73-116 ms | **2.4e-07** | 2.8e-07 |
| 134-238 ms | **1.4e-08** | 1.7e-06 |

The reason: reflected content enters only through the extreme tail of the
kernel, so its effect decays very fast with distance, whereas a shortened
kernel has a systematically different response that applies at full strength
wherever it is used. Perturbing the input slightly costs far less than
changing the filter. The code is kept (`validation/taper_vs_reflect.py`
reproduces this) but reflection remains the default.

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

## Choosing the output rate

**Default is 2048 Hz**, which is the right choice when the destination is
storage-limited. If it is not, 4096 Hz is the better archive and costs no
extra download -- read on.

The download is 20.6 TB and ~5 days *regardless of output rate* -- it is the
same 16 kHz source either way. The rate only changes what you store, and
storage is the cheap axis:

| output | stored | downloaded |
|---|---|---|
| 2048 Hz | 1.17 TB | 20.6 TB / ~5 days |
| **4096 Hz** | **2.33 TB** | 20.6 TB / ~5 days |
| 8192 Hz | 4.67 TB | 20.6 TB / ~5 days |

A 2048 Hz product has its Nyquist at 1024 Hz, so it cannot contain the
1600-2000 Hz band -- which is exactly where GWOSC's own product loses 25.9%
of SNR, and therefore the only reason to re-derive from 16 kHz at all.
Archiving at 2048 Hz spends five days of bandwidth to buy roughly **0.12%**
over simply decimating GWOSC's existing 4 kHz files, which would have cost a
day and a half.

Nothing is lost by archiving higher: cascading 16384 -> 4096 -> 2048 matches a
direct 16384 -> 2048 to **4.2e-8**, so a 2048 Hz set can be generated from the
archive later in a few hours with no re-download. That cascade is only safe
because the 4096 Hz product here is flat to 2000 Hz at -179 dB -- it is
exactly what you cannot do from GWOSC's 4 kHz files.

At 4096 Hz the kernel is also half as long as at 2048 Hz (4093 vs 8183 taps),
so the edge-affected region halves too, to 125 ms.

If storage is tight but not desperate, a reasonable split is 4096 Hz for O4 --
the most sensitive data, where high-frequency content is worth most -- and
2048 Hz for the rest, via two runs with `--runs`:

```bash
python scripts/bulk_reduce.py --dest DEST --rate 4096 --runs O4a,O4b   # 0.96 TB
python scripts/bulk_reduce.py --dest DEST --rate 2048 --runs O3a,O3b,O2,O1  # 0.69 TB
```

## Bulk production

`scripts/bulk_reduce.py` reduces whole observing runs, downloading each 16 kHz
file, writing the reduced version and deleting the original, so the 20.6 TB
source never lands on disk -- peak local footprint is a few GB of scratch.

```bash
python scripts/bulk_reduce.py \
    --dest   /path/to/output \
    --cache  /fast/local/scratch \
    --rate   2048 \
    --workers 10
```

Each `(run, detector)` is an independent sequential stream, so neighbouring
files stay cached as filter context; streams run in parallel, which is what
keeps the network busy. `--runs` selects and orders the runs (default is
`O4a,O4b,O3a,O3b,O2,O1`); `--max-files` limits each stream for a smoke test.

It is **resumable**: valid outputs are skipped, so an interrupted run can be
restarted with the same command, and a partially-populated destination can be
handed to a different machine and continued there.

Expect roughly 20.6 TB downloaded and ~1.2 TB written for the full set at the
default 2048 Hz (~2.3 TB at 4096 Hz).
The job is network-bound -- about 5 days on a 45 MB/s link -- so the only
thing that meaningfully speeds it up is a faster connection.

## Current state

No production output has been kept. The 204 files produced during
development are 2048 Hz and used the current method, but a fresh run from an
empty destination is cleaner; the job is resumable either way.

Settled by measurement, and not worth revisiting without new evidence:

| decision | why |
|---|---|
| source = 16 kHz, not the official 4 kHz | their filter costs 25.9% of SNR above 1600 Hz |
| output rate | 2048 Hz default for storage-limited destinations; 4096 Hz is better where there is room, at no extra download |
| passband 1000/1024 of Nyquist | 2000 Hz flat; ripple 1.8e-9, stopband -173 dB |
| Kaiser FIR, not IIR | flat passband and arbitrary rejection |
| odd reflection at edges | beats zero-padding ~40x and beats kernel tapering up to 100x |
| emit past edges, flag per sample | no science time lost; 0.008% of samples flagged |
| float32, unscaled | quantisation 76 dB below the data; analysis codes apply their own DYN_RANGE_FAC |
| gzip 4 + shuffle | shuffle is the whole win; level and codec changes gain <1% |

## Tests

```bash
pytest
python validation/compare_to_gwosc.py <a 16KHZ file>
```
