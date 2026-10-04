#!/usr/bin/env python3
"""
End-to-end check: inject a compact-binary inspiral into real 16 kHz strain,
reduce the data with this package, and confirm the recovered matched-filter
SNR, arrival time and phase survive the reduction.

The template is generated independently at each sample rate from the same
analytic waveform, so any disagreement is the reduction's fault.
"""
import argparse, os, sys
import h5py, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_sampler.filters import reduce_strain  # noqa: E402

MSUN = 4.925491025543576e-6          # solar mass in seconds (G=c=1)


def taylorf2(f, mc_msun, tc, phic, amp=1.0):
    """Restricted PN inspiral, leading-order phase. Zero outside (0, f_isco)."""
    mc = mc_msun * MSUN
    h = np.zeros(len(f), complex)
    good = f > 0
    psi = (2 * np.pi * f[good] * tc - phic - np.pi / 4
           + (3.0 / 128.0) * (np.pi * mc * f[good]) ** (-5.0 / 3.0))
    h[good] = amp * mc ** (5.0 / 6.0) * f[good] ** (-7.0 / 6.0) * np.exp(1j * psi)
    return h


def welch_psd(x, fs, nseg):
    n = int(nseg * fs)
    w = np.hanning(n)
    segs = [x[i:i + n] * w for i in range(0, len(x) - n + 1, n // 2)]
    p = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], axis=0)
    p *= 2.0 / (fs * (w ** 2).sum())
    return np.fft.rfftfreq(n, 1 / fs), p


def tukey(n, alpha=0.125):
    w = np.ones(n)
    m = int(alpha * n / 2)
    if m > 0:
        e = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m], w[-m:] = e, e[::-1]
    return w


def matched_filter(d, fs, mc, flow, fhigh, psd_seg=16.0, edge_s=8.0):
    """Return (peak SNR, peak time, phase at peak, snr series)."""
    n = len(d)
    w = tukey(n)
    d = d * w                      # suppress leakage from the steep low-f wall
    f = np.fft.rfftfreq(n, 1 / fs)
    pf, pp = welch_psd(d, fs, psd_seg)
    S = np.interp(f, pf, pp)
    band = (f >= flow) & (f <= fhigh)
    S[~band] = np.inf

    h = taylorf2(f, mc, tc=0.0, phic=0.0)
    h[~band] = 0
    df = f[1] - f[0]
    sigsq = 4 * df * np.sum(np.abs(h[band]) ** 2 / S[band])

    dF = np.fft.rfft(d) / fs
    z = 4 * df * n * np.fft.irfft(dF * np.conj(h) / S, n=n)
    zq = 4 * df * n * np.fft.irfft(1j * dF * np.conj(h) / S, n=n)
    snr = np.sqrt(z ** 2 + zq ** 2) / np.sqrt(sigsq)
    guard = int(edge_s * fs)       # ignore filter/window wrap-around at the ends
    search = np.zeros(n, bool)
    search[guard:n - guard] = True
    i = int(np.argmax(np.where(search, snr, 0)))
    return snr[i], i / fs, np.arctan2(zq[i], z[i]), snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file16")
    ap.add_argument("--rate", type=int, default=2048)
    ap.add_argument("--dur", type=float, default=256.0)
    ap.add_argument("--mc", type=float, default=1.2, help="chirp mass (Msun)")
    ap.add_argument("--snr", type=float, default=20.0, help="target network SNR")
    ap.add_argument("--flow", type=float, default=20.0)
    a = ap.parse_args()
    fhigh = 0.9 * a.rate / 2

    with h5py.File(a.file16, "r") as f:
        x = f["strain/Strain"][:]
    ok = np.isfinite(x)
    d = np.diff(np.concatenate(([0], ok.view(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    i = np.argmax(e - s)
    fs = 16384
    n = int(a.dur * fs)
    noise = x[s[i]:s[i] + n].astype(float)
    print(f"{a.dur:.0f} s of real O4a strain at {fs} Hz; "
          f"template Mc={a.mc} Msun, band {a.flow}-{fhigh:.0f} Hz\n")

    # ---- build the injection on the 16 kHz grid ----
    f16 = np.fft.rfftfreq(n, 1 / fs)
    tc = a.dur / 2
    hf = taylorf2(f16, a.mc, tc=tc, phic=0.3)
    hf[(f16 < a.flow) | (f16 > fhigh)] = 0
    ht = np.fft.irfft(hf, n=n) * fs

    # scale to the requested optimal SNR in the 16 kHz data
    pf, pp = welch_psd(noise, fs, 16.0)
    S = np.interp(f16, pf, pp)
    band = (f16 >= a.flow) & (f16 <= fhigh)
    opt = np.sqrt(4 * (f16[1] - f16[0]) *
                  np.sum(np.abs(hf[band]) ** 2 / S[band]))
    ht *= a.snr / opt
    inj = noise + ht

    # ---- pipeline A: matched filter the 16 kHz data directly ----
    sA, tA, pA, _ = matched_filter(inj, fs, a.mc, a.flow, fhigh)
    # ---- pipeline B: reduce with this package, then matched filter ----
    y, edge, _, _ = reduce_strain(inj, a.rate)
    m = np.isfinite(y)
    sB, tB, pB, _ = matched_filter(y[m], a.rate, a.mc, a.flow, fhigh)

    cA, _, _, _ = matched_filter(noise, fs, a.mc, a.flow, fhigh)
    yn, _, _, _ = reduce_strain(noise, a.rate)
    mn = np.isfinite(yn)
    cB, _, _, _ = matched_filter(yn[mn], a.rate, a.mc, a.flow, fhigh)
    print(f"noise-only control: loudest SNR  16 kHz {cA:.2f}   "
          f"{a.rate} Hz {cB:.2f}   (expect ~5)\n")
    print(f"{'pipeline':34} {'SNR':>8} {'t_peak s':>10} {'phase':>8}")
    print(f"{'A  16384 Hz (truth)':34} {sA:8.4f} {tA:10.5f} {pA:8.4f}")
    print(f"{'B  reduced to %d Hz' % a.rate:34} {sB:8.4f} {tB:10.5f} {pB:8.4f}")
    dt = (tB - tA) * 1e6
    print(f"\nSNR ratio B/A      : {sB/sA:.6f}   ({100*(sB/sA-1):+.4f} %)")
    print(f"timing shift       : {dt:+.2f} us  (one sample at {a.rate} Hz = "
          f"{1e6/a.rate:.1f} us)")
    print(f"phase difference   : {np.angle(np.exp(1j*(pB-pA))):+.5f} rad")
    print(f"edge-flagged used  : {int((edge[m]!=0).sum())}")

    ok_snr = abs(sB / sA - 1) < 2e-3
    ok_t = abs(dt) < 1e6 / a.rate
    print(f"\nSNR within 0.2%    : {'PASS' if ok_snr else 'FAIL'}")
    print(f"timing within 1 sample: {'PASS' if ok_t else 'FAIL'}")
    return 0 if (ok_snr and ok_t) else 1


if __name__ == "__main__":
    sys.exit(main())
