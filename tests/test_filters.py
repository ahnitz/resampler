import numpy as np
import pytest
from scipy import signal

from data_sampler.filters import (FS_IN, EDGE_CLEAN, EDGE_PAD, EDGE_EXTRAP,
                                  design, reduce_strain)

RATES = [4096, 2048, 1024]


@pytest.mark.parametrize("rate", RATES)
def test_filter_spec(rate):
    """Passband must be flat and the stopband must actually suppress."""
    taps, pb = design(rate)
    w, H = signal.freqz(taps, worN=60000, fs=FS_IN)
    A = np.abs(H)
    assert np.abs(A[w <= pb] - 1).max() < 1e-7       # GWOSC's own is 1.1e-2
    assert A[w >= rate / 2].max() < 10 ** (-150 / 20)  # GWOSC's own is -44 dB


@pytest.mark.parametrize("rate", RATES)
def test_sine_is_preserved(rate):
    """An in-band sine survives decimation with its amplitude and phase."""
    f0, dur, amp = 137.0, 64, 1e-18
    t = np.arange(dur * FS_IN) / FS_IN
    x = amp * np.sin(2 * np.pi * f0 * t)
    y, edge, _, _ = reduce_strain(x, rate)
    ref = amp * np.sin(2 * np.pi * f0 * np.arange(len(y)) / rate)
    m = (edge == EDGE_CLEAN) & np.isfinite(y)
    assert m.sum() > 0.9 * len(y)
    assert np.abs(y[m] - ref[m]).max() / amp < 1e-6


def test_out_of_band_is_rejected():
    """Power above the output Nyquist must not alias back into the band."""
    dur, amp = 32, 1e-18
    t = np.arange(dur * FS_IN) / FS_IN
    # 3000 Hz would alias to 1096 Hz... and 5000 Hz to 904 Hz at fs_out=2048
    x = amp * (np.sin(2 * np.pi * 3000 * t) + np.sin(2 * np.pi * 5000 * t))
    y, edge, _, _ = reduce_strain(x, 2048)
    m = (edge == EDGE_CLEAN) & np.isfinite(y)
    assert np.abs(y[m]).max() / amp < 1e-6


def test_file_boundary_is_invisible():
    """Splicing two 'files' of continuous data must leave no seam."""
    f0, amp = 211.0, 1e-18
    t = np.arange(3 * 4096 * FS_IN) / FS_IN
    x = amp * np.sin(2 * np.pi * f0 * t)
    y, edge, _, _ = reduce_strain(x, 2048)
    mid = slice(4096 * 2048, 2 * 4096 * 2048)        # the middle 'file'
    assert (edge[mid] == EDGE_CLEAN).all()
    ref = amp * np.sin(2 * np.pi * f0 * np.arange(len(y)) / 2048)
    assert np.abs(y[mid] - ref[mid]).max() / amp < 1e-6


def test_gap_handling_emits_everything():
    """No science time is lost at a real gap, and edges are flagged."""
    amp, dur = 1e-18, 600
    t = np.arange(dur * FS_IN) / FS_IN
    x = amp * np.sin(2 * np.pi * 137.0 * t)
    x[200 * FS_IN:260 * FS_IN] = np.nan              # a real gap
    y, edge, _, _ = reduce_strain(x, 2048, extend_s=0.25)
    covered = np.isfinite(x).reshape(-1, 8).any(axis=1)
    # every covered output sample is emitted, plus flagged ring-out past edges
    assert np.isfinite(y)[covered].all()
    assert np.isfinite(y).sum() > covered.sum()
    assert (edge == EDGE_EXTRAP).any() and (edge == EDGE_PAD).any()


def test_rate_must_divide():
    with pytest.raises(ValueError):
        design(3000)


def test_float32_roundtrip_is_negligible():
    """float32 storage must sit far below the noise it is storing."""
    rng = np.random.default_rng(0)
    x = 1.5e-18 * rng.standard_normal(1 << 20)
    err = x.astype(np.float32).astype(np.float64) - x
    assert err.std() / x.std() < 1e-7


def _chirp(f, mc_msun, tc, n_pad=0):
    MSUN = 4.925491025543576e-6
    mc = mc_msun * MSUN
    h = np.zeros(len(f), complex)
    g = f > 0
    h[g] = (mc ** (5 / 6) * f[g] ** (-7 / 6) *
            np.exp(1j * (2 * np.pi * f[g] * tc +
                         (3 / 128) * (np.pi * mc * f[g]) ** (-5 / 3))))
    return h


@pytest.mark.parametrize("rate", [4096, 2048])
def test_injected_chirp_survives_reduction(rate):
    """An inspiral's arrival time and in-band energy must survive reduction."""
    fs, dur, flow = FS_IN, 64.0, 30.0
    fhigh = 0.9 * rate / 2
    n = int(dur * fs)
    f = np.fft.rfftfreq(n, 1 / fs)

    hf = _chirp(f, 1.5, tc=dur / 2)
    hf[(f < flow) | (f > fhigh)] = 0        # strictly inside the output band
    ht = np.fft.irfft(hf, n=n) * fs
    ht *= 1e-18 / np.abs(ht).max()

    y, edge, _, _ = reduce_strain(ht, rate)
    m = np.isfinite(y) & (edge == EDGE_CLEAN)
    assert m.sum() > 0.9 * len(y)

    # arrival time preserved: compare the analytic ENVELOPE, since argmax of
    # the raw waveform can land on a different cycle peak at each rate
    env16 = np.abs(signal.hilbert(ht))
    envlo = np.abs(signal.hilbert(np.where(m, y, 0.0)))
    t16 = int(np.argmax(env16)) / fs
    tlo = int(np.argmax(envlo)) / rate
    assert abs(tlo - t16) < 2e-3

    # energy of a strictly in-band signal is preserved by decimation.
    # Use every emitted sample: the edge-flagged ones still carry real signal,
    # so excluding them would under-count the chirp's energy.
    fin = np.isfinite(y)
    e16 = np.sum(ht ** 2) / fs
    elo = np.sum(y[fin] ** 2) / rate
    assert abs(elo / e16 - 1) < 1e-3


def test_edge_accuracy_against_known_truth():
    """Quantify accuracy at a real boundary, per edge level, against truth."""
    fs, f0, amp = FS_IN, 137.0, 1e-18
    t = np.arange(300 * fs) / fs
    x = amp * np.sin(2 * np.pi * f0 * t)
    x[:50 * fs] = np.nan                     # a true science boundary at 50 s
    y, edge, _, _ = reduce_strain(x, 2048, extend_s=0.25)
    ref = amp * np.sin(2 * np.pi * f0 * np.arange(len(y)) / 2048)
    fin = np.isfinite(y)
    err = {lvl: (np.abs(y[m] - ref[m]).max() / amp if (m := fin & (edge == lvl)).any()
                 else None) for lvl in (0, 1, 2)}
    assert err[0] < 1e-8          # clean: essentially exact
    assert err[1] < 1e-3          # mirror-padded: still good
    assert err[2] is not None     # extrapolated: emitted and flagged
    # and no science time is thrown away
    covered = np.isfinite(x).reshape(-1, 8).any(axis=1)
    assert fin[covered].all()
