#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lagged Transfer Entropy (TE) with surrogate-based significance.
Discrete (quantile-binned) estimator with Miller-Madow bias correction —
robust for short daily panels. Directional: TE_{Y->X}(tau) = I(X_{t+tau};
Y_t | X_t). Significance via circular-shift surrogates (preserve each series'
autocorrelation, destroy cross-alignment).

Pre-registered binning (no post-hoc tuning):
  Y = wet-day indicator  (rain > 0 mm  -> 1)
  X = tilt-increment indicator (d_tilt > its median -> 1)
Synthetic check: known AR coupling chain must be recovered directionally.
"""
import numpy as np


def _binnen(x, bins):
    x = np.asarray(x, dtype=float)
    qs = np.quantile(x[x == x], np.linspace(0, 1, bins + 1)[1:-1]) if bins > 2 else np.median(x[x == x])
    return np.digitize(x, qs).astype(int)


def _cmi_binary(x, y, z):
    """Conditional mutual information I(X;Y|Z) for binary 0/1 arrays, plug-in
    with Miller-Madow correction (subtract (r_x-1)(r_y-1)r_z/(2N) in nats)."""
    N = len(x)
    cmi = 0.0
    for zv in (0, 1):
        m = z == zv
        nz = m.sum()
        if nz == 0:
            continue
        for xv in (0, 1):
            for yv in (0, 1):
                nxyz = np.sum(m & (x == xv) & (y == yv))
                if nxyz == 0:
                    continue
                nxz = np.sum(m & (x == xv))
                nyz = np.sum(m & (y == yv))
                cmi += (nxyz / N) * np.log((nxyz * nz) / (nxz * nyz))
    # Miller-Madow bias correction: bias ~= (#z-cells)/(2N) = 1/N for binary triples
    cmi -= 1.0 / N
    return max(cmi, 0.0)


def te_binary(y, x, tau, n_surr=400, seed=0):
    """TE_{Y->X} at lag tau. y, x are binary 0/1 arrays of equal length.
    x_t is compared against y_{t-tau} conditioned on x_{t-tau}.
    Returns dict with TE (nats), surrogate mean/std, z, p."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=int)
    n = len(y)
    if n - max(tau, 1) < 50:
        return {"te": np.nan, "z": np.nan, "p": np.nan, "n": n - max(tau, 1)}
    if tau > 0:
        xs, ys, zs = x[tau:], y[:-tau], x[:-tau]
    else:
        xs, ys, zs = x, y, x
    te = _cmi_binary(xs, ys, zs)
    surr = np.empty(n_surr)
    for i in range(n_surr):
        k = rng.integers(1, n)  # circular shift of y
        ys_s = np.roll(y, k)[tau:] if tau > 0 else np.roll(y, k)
        surr[i] = _cmi_binary(xs, ys_s, zs)
    z = (te - surr.mean()) / (surr.std() + 1e-12)
    p = float((surr >= te).mean())
    return {"te": float(te), "surr_mean": float(surr.mean()), "z": float(z),
            "p": p, "n": int(len(xs))}


def te_profile(y, x, lags=range(0, 11), n_surr=400, seed=0):
    """TE_{Y->X} across lags."""
    return {lag: te_binary(y, x, lag, n_surr=n_surr, seed=seed + lag) for lag in lags}


# ------------------------------------------------------------------ synthetic check
def synthetic_validation(n=4000, seed=3):
    """Known answer chain: U -> Y -> X (X depends on lag-1 Y only).
    Expect TE_{Y->X}(1) large & significant, TE_{X->Y}(1) ~ 0, and
    TE_{Y->X}(tau>1) decaying."""
    rng = np.random.default_rng(seed)
    U = rng.integers(0, 2, n)
    Y = np.empty(n, dtype=int)
    X = np.empty(n, dtype=int)
    Y[0], X[0] = 0, 0
    for t in range(1, n):
        Y[t] = int(rng.random() < 0.2 + 0.6 * Y[t - 1])          # self-dynamics
        X[t] = int(rng.random() < 0.15 + 0.55 * Y[t - 1])        # driven by Y only
    fwd = te_profile(Y, X, lags=range(0, 6), n_surr=100, seed=10)
    bwd = te_profile(X, Y, lags=range(0, 6), n_surr=100, seed=20)
    return {"Y_to_X": fwd, "X_to_Y": bwd}
