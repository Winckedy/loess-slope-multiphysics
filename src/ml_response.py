#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interpretable machine-learning layer for event-scale slope response.
Gradient-boosted trees (shallow, strongly regularised) predict the post-event
peak tilt change from event features; validated by LEAVE-ONE-SITE-OUT (LOSO)
against two physically-motivated baselines (persistence delta=0, climatology
train-mean). Attribution via permutation importance and partial dependence.
All models from scikit-learn only — fully reproducible.

Honest-reporting design: a site where the model cannot beat the baselines is
reported as *unpredictable*, not hidden (echoes the paper's honest-negative
ethos; expected for Qingyang).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import cross_val_score

from common import load_panels, event_table, SITES

FEATURES = ["rsum", "dur", "imax", "iint", "api7", "api30", "tilt_base",
            "sin_doy", "cos_doy"]


def build_event_features():
    """Event tables per site with seasonal encodings; returns dict of DataFrames
    plus a pooled frame with site one-hot columns. Adds 'delta_s': site-
    standardised amplitude (delta / site-median |delta|) to remove the
    site-specific tilt-sensitivity scale that confounds cross-site regression."""
    panels = load_panels()
    evs = {}
    for s in SITES:
        e = event_table(panels[s])
        e["site"] = s
        e["sin_doy"] = np.sin(2 * np.pi * e["doy"] / 365.25)
        e["cos_doy"] = np.cos(2 * np.pi * e["doy"] / 365.25)
        evs[s] = e
    pooled = pd.concat(evs.values(), ignore_index=True)
    # edge effects only: first event (no pre-event baseline) / last event
    # (no post-event window) — drop instead of impute
    pooled = pooled.dropna(subset=["tilt_base", "delta"]).reset_index(drop=True)
    scale = pooled.groupby("site")["delta"].transform(lambda x: np.median(np.abs(x)))
    pooled["delta_s"] = pooled["delta"] / scale
    for s in SITES:
        pooled[f"site_{s}"] = (pooled["site"] == s).astype(int)
    feats = FEATURES + [f"site_{s}" for s in SITES]
    return evs, pooled[feats + ["delta", "delta_s", "plus", "site"]]


def _reg_metrics(y, p):
    mae = float(np.mean(np.abs(y - p)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1 - ss_res / ss_tot
    from scipy.stats import spearmanr
    rho = float(spearmanr(y, p).statistic) if len(y) > 3 else np.nan
    return {"MAE": mae, "R2": float(r2), "spearman": rho}


def _make_models():
    reg = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, min_samples_leaf=8, random_state=0)
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=8, random_state=0)
    return reg, clf


def loso_validation(evs, pooled):
    """Leave-one-site-out for classification (plus) and two regression targets:
    delta (raw amplitude) and delta_s (site-standardised amplitude).
    Baselines: persistence (predict 0), climatology (train-mean)."""
    feats = FEATURES + [f"site_{s}" for s in SITES]
    out = {}
    for held in SITES:
        tr = pooled[pooled.site != held]
        te = pooled[pooled.site == held]
        if len(te) < 5 or len(tr) < 20:
            continue
        Xtr, Xte = tr[feats].values, te[feats].values
        ptr, pte = tr["plus"].values, te["plus"].values
        _, clf = _make_models()
        clf.fit(Xtr, ptr)
        proba = clf.predict_proba(Xte)[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(pte, proba))
        except ValueError:
            auc = None
        entry = {
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "AUC": auc,
            "base_rate_test": float(pte.mean()),
            "mean_predicted_proba": float(proba.mean()),
        }
        for tgt in ("delta", "delta_s"):
            reg, _ = _make_models()
            ytr, yte = tr[tgt].values, te[tgt].values
            reg.fit(Xtr, ytr)
            pred = reg.predict(Xte)
            m = _reg_metrics(yte, pred)
            entry[tgt] = {
                **m,
                "baseline_persistence_MAE": float(np.mean(np.abs(yte - 0.0))),
                "baseline_climatology_MAE": float(np.mean(np.abs(yte - ytr.mean()))),
            }
        out[held] = entry
    return out


def within_site_cv(evs, pooled):
    """Per-site 5-fold CV R2 / AUC (in-site predictability)."""
    feats = FEATURES + [f"site_{s}" for s in SITES]
    out = {}
    for s in SITES:
        sub = pooled[pooled.site == s]
        if len(sub) < 20:
            out[s] = {"n": int(len(sub)), "R2_cv": None, "AUC_cv": None}
            continue
        reg, clf = _make_models()
        r2 = cross_val_score(reg, sub[feats].values, sub["delta"].values,
                             cv=5, scoring="r2")
        from sklearn.model_selection import cross_val_score as cvs
        auc = cvs(clf, sub[feats].values, sub["plus"].values, cv=5, scoring="roc_auc")
        out[s] = {"n": int(len(sub)),
                  "R2_cv": [float(r2.mean()), float(r2.std())],
                  "AUC_cv": [float(np.nanmean(auc)), float(np.nanstd(auc))]}
    return out


def attribution(pooled):
    """Permutation importance (AUC scoring) + partial-dependence curves of the
    pooled CLASSIFIER (response occurrence), the target that cross-site
    validation showed to generalise."""
    feats = FEATURES + [f"site_{s}" for s in SITES]
    _, clf = _make_models()
    clf.fit(pooled[feats].values, pooled["plus"].values)
    pi = permutation_importance(clf, pooled[feats].values, pooled["plus"].values,
                                n_repeats=100, random_state=0, scoring="roc_auc")
    imp = sorted(zip(feats, pi.importances_mean), key=lambda t: -t[1])
    top = [f for f, v in imp[:3] if v > 0][:2]
    pdp = {}
    if top:
        disp = PartialDependenceDisplay.from_estimator(
            clf, pooled[feats].values, [feats.index(f) for f in top],
            kind="average", grid_resolution=40, response_method="predict_proba",
            method="brute")
        for ax, f in zip(disp.axes_.ravel(), top):
            line = ax.lines[0]
            pdp[f] = {"grid": line.get_xdata().tolist(),
                      "pdp": line.get_ydata().tolist()}
        import matplotlib.pyplot as plt
        plt.close("all")
    return {"permutation_importance_AUC": [[f, float(v)] for f, v in imp],
            "top_features": [f for f, v in imp[:3]],
            "pdp": pdp}


def run_all():
    evs, pooled = build_event_features()
    res = {"n_events": {s: int(len(evs[s])) for s in SITES},
           "loso": loso_validation(evs, pooled),
           "within_site_cv": within_site_cv(evs, pooled)}
    return res, evs, pooled
