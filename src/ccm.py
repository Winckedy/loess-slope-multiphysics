#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convergent Cross Mapping (CCM) — Sugihara et al. 2012, Science.
Pure-numpy simplex projection; lag-resolved CCM (Ye et al. 2015); random-phase
(FT) surrogates for significance; synthetic validation on unidirectionally
coupled logistic maps.

Direction convention: if Y *drives* X, then X's shadow manifold encodes Y, so
cross-mapping Y FROM X converges to positive skill. For rain->tilt we therefore
cross-map rain from the tilt manifold.
"""
import numpy as np


# ------------------------------------------------------------------ core
def embed(x, E, tau=1):
    """Time-delay embedding; returns (N-(E-1)*tau, E) matrix."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    M = N - (E - 1) * tau
    return np.column_stack([x[i * tau: i * tau + M] for i in range(E)])


class Manifold:
    """Precomputed shadow manifold of x_lib: distances, neighbours, weights are
    built once and reused across many target series (fast surrogate testing)."""

    def __init__(self, x_lib, E, tau=1, exclusion=None):
        self.Mx = embed(x_lib, E, tau)
        self.M = len(self.Mx)
        self.offset = (E - 1) * tau
        self.exclusion = E + 1 if exclusion is None else exclusion
        d2 = np.sum((self.Mx[:, None, :] - self.Mx[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        ii, jj = np.meshgrid(np.arange(self.M), np.arange(self.M), indexing="ij")
        d2[np.abs(ii - jj) < self.exclusion] = np.inf
        k = E + 1
        self.nn = np.argsort(d2, axis=1)[:, :k]
        dn = np.sqrt(np.take_along_axis(d2, self.nn, axis=1))
        w = np.exp(-dn / np.maximum(dn[:, :1], 1e-12))
        self.w = w / w.sum(axis=1, keepdims=True)

    def skill(self, y_target):
        """Pearson r between y and its cross-map from this manifold. y_target is
        aligned to the FULL series; rows self.offset: are used."""
        y = np.asarray(y_target, dtype=float)[self.offset:]
        yhat = (self.w * y[self.nn]).sum(axis=1)
        ok = np.isfinite(yhat) & np.isfinite(y)
        if ok.sum() < 10:
            return np.nan
        return float(np.corrcoef(yhat[ok], y[ok])[0, 1])


def simplex_cross_map(x_lib, y_target, E, tau=1, exclusion=None):
    """Skill (Pearson r) of cross-mapping y_target from x_lib's manifold."""
    return Manifold(x_lib, E, tau, exclusion).skill(y_target)


def ccm_skill_curve(x_lib, y_target, E, tau=1, lib_fracs=(0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
                    n_boot=40, seed=0):
    """Cross-map skill vs library size (mean +- sd over bootstrap subsamples)."""
    rng = np.random.default_rng(seed)
    n = len(x_lib)
    out = []
    for f in lib_fracs:
        L = max(int(round(f * n)), E * tau + k_min(E))
        if L >= n:
            skills = [simplex_cross_map(x_lib, y_target, E, tau)]
        else:
            skills = []
            for _ in range(n_boot):
                idx = np.sort(rng.choice(n, L, replace=False))
                skills.append(simplex_cross_map(x_lib[idx], np.asarray(y_target)[idx], E, tau))
        out.append((L, float(np.nanmean(skills)), float(np.nanstd(skills))))
    return out


def k_min(E):
    return E + 1


def lagged_ccm(x_lib, y_target, E, lags=range(-30, 31), tau=1):
    """Cross-map skill of y(t-lag) from x's present-state manifold, for each lag.
    lag>0: x(t) encodes y from `lag` days earlier (x depends on past y)."""
    y = np.asarray(y_target, dtype=float)
    mf = Manifold(x_lib, E, tau)
    out = {}
    for lag in lags:
        if lag > 0:
            yl = np.concatenate([np.full(lag, np.nan), y[:-lag]])
        elif lag < 0:
            yl = np.concatenate([y[-lag:], np.full(-lag, np.nan)])
        else:
            yl = y.copy()
        out[lag] = mf.skill(yl)
    return out


# ------------------------------------------------------------------ surrogates
def random_phase_surrogates(x, n_surr=200, seed=1):
    """FT (random-phase) surrogates: preserve power spectrum/autocorrelation,
    destroy nonlinear structure and cross-coupling (Theiler et al. 1992)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    xz = (x - x.mean())
    X = np.fft.rfft(xz)
    amp = np.abs(X)
    out = np.empty((n_surr, len(x)))
    for i in range(n_surr):
        ph = rng.uniform(0, 2 * np.pi, size=len(X))
        ph[0] = 0.0
        s = np.fft.irfft(amp * np.exp(1j * ph), n=len(x))
        out[i] = s + x.mean()
    return out


def ccm_significance(x_lib, y_target, E, lags=range(0, 15), n_surr=200, seed=1,
                     lag_window=2):
    """Best-lag skill of real pair vs 95th percentile of surrogate pairs.
    Surrogates are evaluated only in a +-lag_window window around the real best
    lag (speed); the surrogate statistic is the max skill inside that window,
    which is conservative (max over 2*lag_window+1 lags)."""
    real = lagged_ccm(x_lib, y_target, E, lags=lags)
    best_lag = max(real, key=lambda L: (real[L] if np.isfinite(real[L]) else -np.inf))
    best_skill = real[best_lag]
    lo = max(min(lags), best_lag - lag_window)
    hi = min(max(lags), best_lag + lag_window)
    wl = list(range(lo, hi + 1))
    mf = Manifold(x_lib, E)  # manifold built once, reused for all surrogates
    surr = random_phase_surrogates(y_target, n_surr, seed)
    ss = []
    for i in range(n_surr):
        ss.append(max(mf.skill(_shift(y_target_len := len(surr[i]), surr[i], L))
                      for L in wl))
    ss = np.array(ss)
    z = (best_skill - ss.mean()) / (ss.std() + 1e-12)
    p = float((ss >= best_skill).mean())
    return {"best_lag": int(best_lag), "skill": float(best_skill),
            "surr_mean": float(ss.mean()), "surr_p95": float(np.percentile(ss, 95)),
            "z": float(z), "p": p}


def _shift(n, y, lag):
    """yl[t] = y[t-lag] padded with NaN (same convention as lagged_ccm)."""
    if lag > 0:
        return np.concatenate([np.full(lag, np.nan), y[:-lag]])
    return y.copy()


# ------------------------------------------------------------------ synthetic check
def _logistic_pair(n=3000, c_xy=0.32, c_yx=0.0, seed=7):
    """Unidirectionally coupled logistic maps: y drives x when c_xy>0.
    x_{t+1} = x(3.8-3.8x) + c_xy*y ; y_{t+1} = y(3.5-3.5y) + c_yx*x
    Expect: cross-map x FROM y converges (y encoded in x); reverse stays ~0
    when c_yx=0 (classic Sugihara Fig.1 asymmetry).
    """
    rng = np.random.default_rng(seed)
    x, y = 0.4, 0.2
    xs, ys = np.empty(n), np.empty(n)
    for t in range(n):
        xs[t], ys[t] = x, y
        x = np.clip(x * (3.8 - 3.8 * x) + c_xy * y, 1e-6, 1 - 1e-6)
        y = np.clip(y * (3.5 - 3.5 * y) + c_yx * x, 1e-6, 1 - 1e-6)
    return xs, ys


def synthetic_validation(n=3000, c_xy=0.02, c_yx=0.0):
    """Known answer: y->x weak coupling only (weak coupling is required for the
    asymmetry: at very strong coupling the maps synchronize and CCM turns
    symmetric). Return skill curves both directions; the true direction must
    CONVERGE upward with library size while the reverse stays low."""
    xs, ys = _logistic_pair(n, c_xy, c_yx)
    curve_y_from_x = ccm_skill_curve(xs, ys, E=3, lib_fracs=(0.1, 0.3, 0.5, 0.8, 1.0),
                                     n_boot=20)  # cross-map y from x
    curve_x_from_y = ccm_skill_curve(ys, xs, E=3, lib_fracs=(0.1, 0.3, 0.5, 0.8, 1.0),
                                     n_boot=20)
    return {"y_from_x": curve_y_from_x, "x_from_y": curve_x_from_y}
