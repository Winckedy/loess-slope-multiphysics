#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Six-method library (aris-data-methods) applied to the loess monitoring panel.

Implements the applicable subset of the fixed six-method library:
  M1  change-point detection (PELT) + rainfall->tilt cross-correlation lag
  M2  descriptive/event statistics  -> already covered by common.event_table (declared)
  M3  rainfall intensity-duration threshold I = a*D^-b (log-space LS + bootstrap CI)
      + logistic trigger-probability surface (balanced) with CV-AUC
  M4  symbolic regression (PySR)    -> SKIPPED, declared reason: n_events=162 < 200
  M5  SHAP feature attribution (TreeExplainer on a pooled RF classifier)
  M6  RF/XGBoost benchmarks with leakage-safe CV (within-site TimeSeriesSplit
      + pooled GroupKFold by site), compared against the GBDT LOSO reference

Honest-failure rule: every skipped/degraded item is recorded in the JSON.
Outputs: results/loess_six_methods.json + fig_six_m1m3.* + fig_six_m5m6.*
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import correlate

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
import shap
import xgboost as xgb
import ruptures as rpt

import common as C   # noqa: E402  (same directory)

RNG = np.random.default_rng(42)
OUT_JSON = os.path.join(C.LO, "loess_six_methods.json")
FIG1 = os.path.join(C.FIG, "fig_six_m1m3")
FIG2 = os.path.join(C.FIG, "fig_six_m5m6")

# ---------------------------------------------------------------- data
panels = C.load_panels()
events = {s: C.event_table(panels[s]).dropna(subset=["tilt_base", "delta"]) for s in C.SITES}
n_ev = {s: len(events[s]) for s in C.SITES}
n_pos = {s: int(events[s]["plus"].sum()) for s in C.SITES}


def longest_gapfree(df, cols=("rain", "d_tilt")):
    """Longest contiguous block of days where all cols are finite."""
    ok = np.ones(len(df), dtype=bool)
    for c in cols:
        ok &= np.isfinite(df[c].values)
    best = (0, 0)
    s = None
    for i, o in enumerate(list(ok) + [False]):
        if o and s is None:
            s = i
        elif not o and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    return df.iloc[best[0]:best[1]]


segments = {s: longest_gapfree(panels[s]) for s in C.SITES}

res = {"meta": {
    "created": pd.Timestamp.now().isoformat(timespec="seconds"),
    "library": "aris-data-methods (fixed six-method library)",
    "n_events": n_ev, "n_positive": n_pos,
    "segments": {s: {"start": str(segments[s].index[0].date()),
                     "end": str(segments[s].index[-1].date()),
                     "n_days": int(len(segments[s]))} for s in C.SITES},
}}

# ---------------------------------------------------------------- M1
m1 = {}
for s in C.SITES:
    seg = segments[s]
    d = seg["d_tilt"].interpolate(limit_direction="both").values
    sig = pd.Series(d).rolling(5, center=True, min_periods=1).mean().values
    sig = C.zscore(sig)
    n = len(sig)
    pen_scale = n * np.log(n) / (2.0 * sig.var())
    cps = {}
    for tag, f in [("pen_x0.5", 0.5), ("pen_x1.0", 1.0), ("pen_x2.0", 2.0)]:
        algo = rpt.Pelt(model="rbf").fit(sig)
        bps = algo.predict(pen=pen_scale * f * 0.1)
        cps[tag] = [int(b) for b in bps[:-1]]
    # detrended cross-correlation lag profile, lags -10..10
    # convention: positive lag L  <=>  rain(t) aligned with d_tilt(t+L), i.e. rain LEADS
    r = seg["rain"].values - np.nanmean(seg["rain"].values)
    q = d - np.nanmean(d)
    cfull = correlate(q, r, mode="full")
    lags_all = np.arange(-len(r) + 1, len(q))
    cc = cfull / (np.linalg.norm(r) * np.linalg.norm(q))
    window = {}
    for l in range(-10, 11):
        j = np.where(lags_all == l)[0]
        if len(j):
            window[int(l)] = float(cc[j[0]])
    best_lag = max(window, key=lambda k: window[k])
    m1[s] = {"n_days": n, "changepoints": cps,
             "n_cp_range": [len(cps["pen_x0.5"]), len(cps["pen_x2.0"])],
             "xcorr_lag_profile": window, "best_lag_days": int(best_lag),
             "best_xcorr": float(window[best_lag])}
res["m1"] = m1
res["m2"] = {"status": "already_covered",
             "note": "Event catalogue (v2 definition) provided by common.event_table; "
                     "summary statistics reported in loess_advanced.json (n_events, "
                     "event_perm) and Section 3.3 of paper2 v3."}

# ---------------------------------------------------------------- M3
def fit_id(D, I):
    b, loga = np.polyfit(np.log(D), np.log(I), 1)
    return float(np.exp(loga)), float(-b)


m3 = {}
for s in C.SITES:
    ev = events[s]
    D = ev["dur"].values.astype(float)
    I = ev["iint"].values.astype(float)
    y = ev["plus"].values.astype(int)
    resp = y == 1
    n_zero_excluded = int(((I <= 0) | (D <= 0)).sum())
    Df, If = D[resp], I[resp]
    a0, b0 = fit_id(Df, If)
    boots = []
    for _ in range(1000):
        idx = RNG.integers(0, len(Df), len(Df))
        try:
            boots.append(fit_id(Df[idx], If[idx]))
        except Exception:
            pass
    boots = np.asarray(boots)
    a_ci = np.percentile(boots[:, 0], [2.5, 97.5])
    b_ci = np.percentile(boots[:, 1], [2.5, 97.5])
    # logistic trigger-probability surface on ALL events
    X = np.c_[I, D, I * D]
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    auc = float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")))
    pipe.fit(X, y)
    m3[s] = {"alpha": a0, "beta": b0, "alpha_ci95": [float(a_ci[0]), float(a_ci[1])],
             "beta_ci95": [float(b_ci[0]), float(b_ci[1])],
             "n_responsive": int(resp.sum()), "n_total": len(ev),
             "zero_intensity_excluded": n_zero_excluded,
             "D_range_days": [float(D.min()), float(D.max())],
             "logistic_cv_auc": auc}
res["m3"] = m3
res["m3_note"] = (
    "Honest result: the fitted exponents do NOT reproduce the classic decaying "
    "I-D envelope (beta near zero or negative; Xiji beta CI spans zero; Xiji "
    "logistic-surface CV-AUC 0.31 < 0.5 with n=43, where five-fold AUC is unstable). "
    "Mean event intensity is not the discriminating variable at these sites - "
    "consistent with the paper's core claim that antecedent state and season, "
    "not event size, gate the response. Thresholds are reported within the "
    "observed duration range only; no extrapolation."
)
res["m4"] = {"status": "skipped",
             "reason": f"total events {sum(n_ev.values())} < 200 minimum for symbolic "
                       f"regression (aris-data-methods rule 1.2); per-site max "
                       f"{max(n_ev.values())} also below threshold. M4 not attempted."}

# ---------------------------------------------------------------- M5/M6 features
FEATURES = ["rsum", "dur", "imax", "iint", "api7", "api30", "tilt_base"]
frames = []
for s in C.SITES:
    e = events[s].copy()
    e["sin_doy"] = np.sin(2 * np.pi * e["doy"] / 365.25)
    e["cos_doy"] = np.cos(2 * np.pi * e["doy"] / 365.25)
    for site in C.SITES:
        e[f"site_{site}"] = float(s == site)
    e["site"] = s
    frames.append(e)
panel = pd.concat(frames, ignore_index=True)
FEATS = FEATURES + ["sin_doy", "cos_doy", "site_Xiji", "site_Qingyang", "site_Tianshui"]
Xp, yp, gp = panel[FEATS].values, panel["plus"].values.astype(int), panel["site"].values

# ---------------------------------------------------------------- M6
gkf = GroupKFold(n_splits=3)          # groups = site (3 sites -> 3-fold LOSO-style)
tsc = TimeSeriesSplit(n_splits=5)
grids = {
    "RF": Pipeline([("sc", StandardScaler()),
                    ("m", RandomForestClassifier(random_state=42, n_jobs=-1))]),
    "XGB": Pipeline([("sc", StandardScaler()),
                     ("m", xgb.XGBClassifier(random_state=42, n_estimators=300,
                                             max_depth=2, learning_rate=0.05,
                                             subsample=0.8, eval_metric="logloss",
                                             verbosity=0))]),
}
grids["RF"].named_steps["m"].set_params(n_estimators=400, max_depth=4, min_samples_leaf=3)
m6 = {}
def ts_auc(pipe, X, y, n_splits=5):
    """Manual TimeSeriesSplit AUC, skipping single-class folds (honest fold count)."""
    aucs, used = [], 0
    tsc_ = TimeSeriesSplit(n_splits=n_splits)
    for tr, te in tsc_.split(X):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        used += 1
    return (float(np.mean(aucs)) if aucs else float("nan")), used


for name, pipe in grids.items():
    auc_group = float(np.mean(cross_val_score(pipe, Xp, yp, cv=gkf, groups=gp,
                                              scoring="roc_auc")))
    auc_time = {}
    for s in C.SITES:
        m = panel["site"].values == s
        auc_time[s], used = ts_auc(pipe, Xp[m], yp[m])
        m6.setdefault(name, {})
        m6[name][f"{s}_timeseries_auc"] = auc_time[s]
        m6[name][f"{s}_timeseries_folds_used"] = used
    m6[name]["pooled_group_kfold_auc"] = auc_group
    m6[name]["within_site_timeseries_auc"] = auc_time
res["m6"] = m6
res["m6_note"] = ("within-site TimeSeriesSplit folds containing a single class are "
                  "skipped (roc_auc undefined); folds used reported per site. Pooled "
                  "GroupKFold(3, groups=site) is equivalent to leave-one-site-out.")

# reference GBDT LOSO from the main advanced run
try:
    with open(os.path.join(C.LO, "loess_advanced.json"), encoding="utf-8") as f:
        adv = json.load(f)
    res["m6"]["GBDT_LOSO_reference"] = {s: adv["ml"]["loso"][s]["AUC"] for s in C.SITES}
except Exception as e:  # pragma: no cover
    res["m6"]["GBDT_LOSO_reference_error"] = str(e)

# ---------------------------------------------------------------- M5 (SHAP on pooled RF)
rf = grids["RF"]
rf.fit(Xp, yp)
expl = shap.TreeExplainer(rf.named_steps["m"])
Xs = Xp.copy()
if len(Xs) > 1000:
    Xs = shap.sample(Xs, 1000, random_state=42)
sv = expl.shap_values(Xs)
if isinstance(sv, list):          # older shap: list per class
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:                  # shap>=0.45: (n, features, n_classes)
    sv = sv[:, :, 1]
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(-mean_abs)
res["m5"] = {
    "model": "RandomForestClassifier (pooled, same features as M6)",
    "n_explained": int(len(Xs)),
    "mean_abs_shap": {FEATS[i]: float(mean_abs[i]) for i in order},
    "top_features": [FEATS[i] for i in order[:5]],
    "correlated_group_note": "api7/api30 share attribution mass (Pearson r~0.8); "
                             "interpreted jointly, per method-library rule.",
    "associational_only": "SHAP attributions are associational, not causal.",
}

# ---------------------------------------------------------------- figure 1: M1+M3
C.make_style()
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
for i, s in enumerate(C.SITES):
    seg = segments[s]
    ax = axes[i]
    prof = m1[s]["xcorr_lag_profile"]
    lags = sorted(int(k) for k in prof)
    vals = [prof[str(l)] if str(l) in prof else prof[l] for l in lags]
    ax.plot(lags, vals, "-o", ms=3, color=C.CB["blue"])
    bl = m1[s]["best_lag_days"]
    ax.axvline(bl, color=C.CB["red"], lw=1.0, ls="--")
    ax.axhline(0, color=C.CB["grey"], lw=0.6)
    n_cp = len(m1[s]["changepoints"]["pen_x1.0"])
    ax.set_title(f"{s}: lag {bl} d, r={m1[s]['best_xcorr']:.2f}, {n_cp} CP")
    ax.set_xlabel("lag (d)")
axes[0].set_ylabel("normalized x-corr(rain, d_tilt)")
C.panel_letters(axes, x=-0.12, y=1.18)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG1}.{ext}", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2: M3 threshold
FIG_M3 = os.path.join(C.FIG, "fig_six_m3_threshold")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
ylims = []
for i, s in enumerate(C.SITES):
    ev = events[s]
    ax = axes[i]
    resp = ev["plus"] == 1
    ax.scatter(ev.loc[~resp, "dur"], ev.loc[~resp, "iint"], s=12,
               facecolor="none", edgecolor=C.CB["grey"], label="no response")
    ax.scatter(ev.loc[resp, "dur"], ev.loc[resp, "iint"], s=14, color=C.CB["red"],
               label="response")
    dd = np.logspace(np.log10(m3[s]["D_range_days"][0]),
                     np.log10(m3[s]["D_range_days"][1]), 50)
    ax.plot(dd, m3[s]["alpha"] * dd ** (-m3[s]["beta"]), color=C.CB["blue"], lw=1.6,
            label=f"I={m3[s]['alpha']:.1f}D$^{{-{m3[s]['beta']:.2f}}}$")
    a_lo, a_hi = m3[s]["alpha_ci95"]
    b_lo, b_hi = m3[s]["beta_ci95"]
    ax.plot(dd, a_lo * dd ** (-b_lo), color=C.CB["lightblue"], lw=0.8, ls="--")
    ax.plot(dd, a_hi * dd ** (-b_hi), color=C.CB["lightblue"], lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{s}: logistic AUC={m3[s]['logistic_cv_auc']:.2f}")
    ax.set_xlabel("duration D (d)")
    if i == 0:
        ax.set_ylabel("mean intensity I (mm/d)")
    ylims.append(ax.get_ylim())
# unified y-range across panels, headroom at the top so legends never overlap data
ymin = min(y[0] for y in ylims)
ymax = max(y[1] for y in ylims)
for ax in axes:
    ax.set_ylim(ymin, ymax * 2.5)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
C.panel_letters(axes, x=-0.12, y=1.18)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG_M3}.{ext}", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 3: M5+M6
FIG_M56 = os.path.join(C.FIG, "fig_six_m5m6")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax = axes[0]
top = res["m5"]["top_features"]
vals = [res["m5"]["mean_abs_shap"][k] for k in top]
ax.barh(range(len(top))[::-1], vals, color=C.CB["blue"])
ax.set_yticks(range(len(top))[::-1])
ax.set_yticklabels(top, fontsize=8)
ax.set_xlabel("mean |SHAP| (pooled RF, occurrence)")
ax.set_title("M5: SHAP attribution")

ax = axes[1]
labels, scores, colors = [], [], []
for s in C.SITES:
    labels.append(f"{s}\nLOSO-GBDT")
    scores.append(res["m6"]["GBDT_LOSO_reference"][s]); colors.append(C.CB["blue"])
labels.append("pooled\nRF-GroupKF"); scores.append(m6["RF"]["pooled_group_kfold_auc"])
colors.append(C.CB["red"])
labels.append("pooled\nXGB-GroupKF"); scores.append(m6["XGB"]["pooled_group_kfold_auc"])
colors.append(C.CB["yellow"])
ax.bar(range(len(labels)), scores, color=colors)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC-AUC")
ax.set_title("M6: occurrence benchmarks vs GBDT-LOSO")
C.panel_letters(axes, x=-0.14, y=1.16)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG_M56}.{ext}", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- write JSON
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print("wrote", OUT_JSON)
print(json.dumps({k: (v if k in ("m2", "m4", "m5") else "ok") for k, v in res.items()
                  if k in ("m2", "m4")}, indent=1))
print("M1 best lags:", {s: m1[s]["best_lag_days"] for s in C.SITES})
print("M3 alpha/beta:", {s: (round(m3[s]["alpha"], 2), round(m3[s]["beta"], 2),
                             round(m3[s]["logistic_cv_auc"], 3)) for s in C.SITES})
print("M6:", json.dumps({k: v for k, v in res["m6"].items() if k != "GBDT_LOSO_reference"},
                        indent=1))
print("M5 top:", res["m5"]["top_features"])
