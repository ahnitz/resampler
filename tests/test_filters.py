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
