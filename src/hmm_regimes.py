#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-state Gaussian Hidden Markov Model (Baum-Welch + Viterbi), pure numpy.
Formalises the paper's 'intermittent coupling' as an explicit stochastic
process: hidden states = {quiescent, hydro-mechanically active}, emissions =
[d_tilt (z), sqrt(rain) (z)] with diagonal covariance. Outputs transition
matrix, dwell times, state occupancy, Viterbi path, and state-conditional
response statistics. Multiple random restarts against local optima.

Synthetic check: a 2-state chain with known transition matrix and emission
separation must be recovered (|a_ij err| <= 0.03, Viterbi accuracy >= 0.85).
"""
import numpy as np


def _log_emissions(obs, means, covs):
    """Log Gaussian emission densities, diagonal covariance. (T, K)"""
    T, D = obs.shape
    K = means.shape[0]
    logB = np.empty((T, K))
    for k in range(K):
        d = obs - means[k]
        prec = 1.0 / covs[k]
        logdet = np.sum(np.log(covs[k]))
        logB[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet +
                             np.sum(d * d * prec, axis=1))
    return logB


def _forward_backward(logB, logA, logpi):
    """Log-space forward-backward. Returns (logalpha, logbeta) each (T,K)."""
    T, K = logB.shape
    logalpha = np.empty((T, K))
    logalpha[0] = logpi + logB[0]
    for t in range(1, T):
        m = logalpha[t - 1][:, None] + logA  # (from i, to j)
        logalpha[t] = np.logaddexp.reduce(m, axis=0) + logB[t]
    logbeta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        m = logA + (logB[t + 1] + logbeta[t + 1])[None, :]
        logbeta[t] = np.logaddexp.reduce(m, axis=1)
    return logalpha, logbeta


def _viterbi(logB, logA, logpi):
    T, K = logB.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = logpi + logB[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + logA
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = np.max(cand, axis=0) + logB[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path


def fit_hmm(obs, K=2, n_init=20, max_iter=300, tol=1e-5, seed=0):
    """Baum-Welch with random restarts; returns best model dict.
    Numerical guards: gamma floored against underflow, covariance floored at
    1e-4 x data variance, degenerate restarts (state weight collapse) skipped."""
    obs = np.asarray(obs, dtype=float)
    T, D = obs.shape
    var_floor = np.maximum(1e-4 * obs.var(axis=0), 1e-9)
    rng = np.random.default_rng(seed)
    best = None
    for init in range(n_init):
        means = obs[rng.choice(T, K, replace=False)] + \
            rng.normal(0, 0.1, (K, D))
        covs = np.tile(obs.var(axis=0) * 0.5 + var_floor, (K, 1))
        A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(A, 0.9)
        pi = np.full(K, 1.0 / K)
        prev_ll = -np.inf
        ok = True
        for it in range(max_iter):
            logA = np.log(A + 1e-12)
            logpi = np.log(pi + 1e-12)
            logB = _log_emissions(obs, means, covs)
            logalpha, logbeta = _forward_backward(logB, logA, logpi)
            # log P(obs): sum over terminal states of logalpha
            ll = float(np.logaddexp.reduce(logalpha[-1]))
            if not np.isfinite(ll):
                ok = False
                break
            loggamma = logalpha + logbeta
            gamma = np.exp(np.clip(loggamma - ll, -700, 0.0))
            # vectorised xi: xi[t,i,j] = alpha[t,i] A[i,j] B[t+1,j] beta[t+1,j] / P(obs)
            logX = (logalpha[:-1] - ll)[:, :, None] + logA[None, :, :] + \
                (logB[1:] + logbeta[1:])[:, None, :]
            xi_sum = np.exp(np.clip(logX, -700, 0.0)).sum(axis=0)
            # M-step
            w = gamma.sum(axis=0)
            if w.min() < 1e-6 * T:  # state collapsed -> retry with new init
                ok = False
                break
            means = (gamma[:, :, None] * obs[:, None, :]).sum(axis=0) / w[:, None]
            for k in range(K):
                d = obs - means[k]
                covs[k] = np.maximum(
                    np.sum((gamma[:, k][:, None] * d * d), axis=0) / w[k],
                    var_floor)
            A = xi_sum / (gamma[:-1].sum(axis=0)[:, None] + 1e-9)
            A = np.clip(A, 1e-9, 1.0)
            A /= A.sum(axis=1, keepdims=True)
            pi = gamma[0] / gamma[0].sum()
            if it > 10 and ll - prev_ll < tol * max(abs(prev_ll), 1.0):
                break
            prev_ll = ll
        if ok and best is not None and prev_ll <= best["ll"]:
            continue
        if not ok:
            continue
        logB = _log_emissions(obs, means, covs)
        path = _viterbi(logB, np.log(A + 1e-12), np.log(pi + 1e-12))
        if best is None or prev_ll > best["ll"]:
            best = {"ll": float(prev_ll), "means": means, "covs": covs,
                    "A": A, "pi": pi, "path": path}
    if best is None:  # fall back to a safe default so callers never crash
        raise RuntimeError("HMM fit failed for all restarts")
    return best


def dwell_times(A):
    """Expected dwell time per state: 1/(1-a_ii)."""
    return 1.0 / np.clip(1.0 - np.diag(A), 1e-9, None)


def summarize(model, obs_names=("d_tilt", "sqrt_rain")):
    A = model["A"]
    path = model["path"]
    # label: 'active' = state with higher sqrt-rain mean
    k_active = int(np.argmax(model["means"][:, 1]))
    order = [k_active, 1 - k_active] if model["means"].shape[0] == 2 else list(range(len(A)))
    return {
        "transition_matrix": A.tolist(),
        "dwell_times": dwell_times(A).tolist(),
        "active_state": k_active,
        "occupancy_active": float((path == k_active).mean()),
        "state_means": {obs_names[d]: [float(model["means"][k, d]) for k in range(len(A))]
                        for d in range(model["means"].shape[1])},
        "state_sd": {obs_names[d]: [float(np.sqrt(model["covs"][k, d])) for k in range(len(A))]
                     for d in range(model["means"].shape[1])},
        "path_active": (path == k_active).astype(int).tolist(),
    }


# ------------------------------------------------------------------ synthetic check
def synthetic_validation(n=3000, seed=11):
    """Known 2-state chain: state 0 (quiescent) dwell ~ 20 d, state 1 (active)
    dwell ~ 12.5 d; emissions well separated. Recover A and the state path."""
    rng = np.random.default_rng(seed)
    A_true = np.array([[0.95, 0.05], [0.08, 0.92]])
    s = np.empty(n, dtype=int)
    s[0] = 0
    for t in range(1, n):
        s[t] = rng.choice(2, p=A_true[s[t - 1]])
    obs = np.column_stack([
        rng.normal(0.0, 0.5, n) * (s == 0) + rng.normal(1.6, 0.9, n) * (s == 1),
        rng.normal(0.0, 0.4, n) * (s == 0) + rng.normal(1.2, 0.7, n) * (s == 1),
    ])
    m = fit_hmm(obs, K=2, n_init=10, seed=seed)
    A_est = m["A"]
    # align states: state with higher mean of dim-1 is 'active'
    if m["means"][0, 1] > m["means"][1, 1]:
        perm = [1, 0]
    else:
        perm = [0, 1]
    A_aligned = A_est[np.ix_(perm, perm)]
    aligned_path = np.array([perm[k] for k in m["path"]])
    acc = float(np.mean(aligned_path == s))
    err = float(np.abs(A_aligned - A_true).max())
    return {"max_abs_A_err": err, "viterbi_accuracy": acc,
            "A_est_aligned": A_aligned.tolist(), "A_true": A_true.tolist()}
