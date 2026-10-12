#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wavelet coherence (Morlet) with AR(1) Monte-Carlo significance.
Follows Torrence & Compo (1998) and Grinsted et al. (2004): continuous wavelet
transform via FFT, cross-wavelet coherence with time/scale smoothing, cone of
influence, and a red-noise null built from AR(1) surrogate pairs matched to
each series' lag-1 autocorrelation and variance. The SAME smoothing is applied
to real and surrogate pairs, so the Monte-Carlo null is self-consistent.

Synthetic check: a pair coupled only inside a known time-frequency box must be
flagged there (inside 95% contour) and almost nowhere else.
"""
import numpy as np


def _morlet_ft(k, scale, omega0=6.0, dt=1.0):
    """Fourier-domain Morlet (Torrence & Compo eq. 6)."""
    k = k * dt
    return np.sqrt(2 * np.pi * scale * dt) * np.pi ** (-0.25) * \
        np.exp(-((scale * k - omega0) ** 2) / 2.0) * (k > 0)


def cwt(x, scales, omega0=6.0, dt=1.0):
    """Continuous wavelet transform via FFT. Returns (n_scales, n) complex."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    npad = int(2 ** np.ceil(np.log2(n)))
    X = np.fft.rfft(x - x.mean(), npad)
    k = 2 * np.pi * np.fft.rfftfreq(npad, d=dt)
    W = np.empty((len(scales), n), dtype=complex)  # buffer at original length n
    for i, s in enumerate(scales):
        W[i] = np.fft.irfft(X * _morlet_ft(k, s, omega0, dt), npad)[:n]
    return W


def _smooth(A, scales):
    """Time smoothing: per-scale Gaussian, sigma = scale/2.5.
    Scale smoothing: 3-scale running mean. Applied identically to real and
    surrogate coherences (self-consistent null)."""
    n = A.shape[1]
    out = np.empty_like(A)
    for j, s in enumerate(scales):
        sigma = max(s / 2.5, 1.0)
        half = int(np.ceil(3 * sigma))
        t = np.arange(-half, half + 1)
        g = np.exp(-(t ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        full = np.convolve(A[j], g, mode="full")
        start = (len(g) - 1) // 2  # centered crop, robust when kernel > signal
        out[j] = full[start:start + n]
    out = np.stack([out[max(0, j - 1):min(A.shape[0], j + 2)].mean(axis=0)
                    for j in range(A.shape[0])])
    return out


def wavelet_coherence(x, y, dt=1.0, omega0=6.0, n_scales=40, s_min=2.0, s_max=None):
    """Cross-wavelet coherence. Returns (scales, wtc[n_scales, n])."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if s_max is None:
        s_max = n / (2 * 1.03)  # Fourier period ~ 1.03*scale for Morlet
    scales = np.geomspace(s_min, max(s_max, s_min * 2), n_scales)
    Wx = cwt(x, scales, omega0, dt)
    Wy = cwt(y, scales, omega0, dt)
    Sxy = _smooth(Wx * np.conj(Wy), scales)
    Sxx = _smooth((Wx * np.conj(Wx)).real.astype(complex), scales)
    Syy = _smooth((Wy * np.conj(Wy)).real.astype(complex), scales)
    wtc = np.abs(Sxy) ** 2 / (Sxx.real * Syy.real + 1e-12)
    return scales, wtc


def coi_mask(n, scales, dt=1.0):
    """True where INSIDE the cone of influence (edge-affected)."""
    t = np.arange(n)
    mask = np.zeros((len(scales), n), dtype=bool)
    for j, s in enumerate(scales):
        tau = np.sqrt(2) * s  # Morlet e-folding time
        mask[j] = (t < tau) | (t > n - 1 - tau)
    return mask


def ar1_surrogates(x, n_surr=300, seed=0):
    """AR(1) surrogates matched to lag-1 autocorrelation and variance."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    a = max(float(np.corrcoef(xc[:-1], xc[1:])[0, 1]), 0.0)
    eps_sd = xc.std() * np.sqrt(max(1 - a ** 2, 1e-9))
    out = np.empty((n_surr, len(x)))
    for i in range(n_surr):
        e = rng.normal(0, eps_sd, len(x))
        s = np.empty(len(x))
        s[0] = 0.0
        for t in range(1, len(x)):
            s[t] = a * s[t - 1] + e[t]
        out[i] = s
    return out


def wtc_significance(x, y, n_surr=300, seed=0, dt=1.0, **kw):
    """WTC of the real pair + scale-wise 95th-percentile null from independent
    AR(1) surrogate pairs. Returns (scales, wtc, p95)."""
    scales, wtc = wavelet_coherence(x, y, dt=dt, **kw)
    sx = ar1_surrogates(x, n_surr, seed)
    sy = ar1_surrogates(y, n_surr, seed + 1)
    null = np.empty((n_surr, len(scales), len(x)))
    for i in range(n_surr):
        _, w = wavelet_coherence(sx[i], sy[i], dt=dt, **kw)
        null[i] = w
    p95 = np.percentile(null, 95, axis=0)
    return scales, wtc, p95


# ------------------------------------------------------------------ synthetic check
def synthetic_validation(n=1024, seed=5, n_surr=100):
    """y = AR(1) + 0.8*driver only during t in [400,600) at period 16 d.
    Coupling to x = sin(16d) windowed. The 95% contour must flag the planted
    box (Fourier period 12-24 d, t 380-620) and little else outside the COI."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e1, e2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    x = np.empty(n)
    y = np.empty(n)
    x[0] = y[0] = 0.0
    for i in range(1, n):
        x[i] = 0.85 * x[i - 1] + e1[i]
        y[i] = 0.85 * y[i - 1] + e2[i]
    driver = np.sin(2 * np.pi * t / 16.0) * ((t >= 400) & (t < 640))
    x = x + 2.0 * driver
    y = y + 2.0 * driver  # shared component inside the window only
    scales, wtc, p95 = wtc_significance(x, y, n_surr=n_surr, seed=seed)
    mask = coi_mask(n, scales)
    sig = (wtc > p95) & (~mask)
    box = np.zeros_like(sig)
    for j, s in enumerate(scales):
        if 12 <= 1.03 * s <= 24:
            box[j, 390:650] = True
    inside = float(sig[box].mean()) if box.any() else np.nan
    outside = float(sig[(~box) & (~mask)].mean())
    return {"inside_box_hit_rate": inside, "outside_false_rate": outside}
