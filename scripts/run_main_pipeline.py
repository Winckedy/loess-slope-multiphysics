#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Master runner for the advanced (v3) analysis layer.
Five-method ladder: CCM causality, lagged transfer entropy, wavelet coherence,
2-state HMM regimes, interpretable ML on events. Every method is first
validated against synthetic data with a known answer (results stored in
JSON 'validation'). Real-data outputs: results/loess_advanced.json,
results/advanced_aux.npz (state paths / matrices for figures),
4 paper figures (fixed project palette).

Pre-registered choices (declared, not tuned post hoc):
  * CCM/TE/WTC/HMM use the LONGEST CONTIGUOUS segment per site with both
    series finite (Qingyang tilt has a 48-d gap; Tianshui rain a 97-d gap).
  * CCM: E=3, tau=1; primary pair = cross-map rain from the d-tilt manifold;
    reverse direction for asymmetry; 200 random-phase surrogates, +-2 lag
    window around best lag.
  * TE: binary wet-day (rain>0) -> binary above-median daily tilt increment;
    lags 0-10; 400 circular-shift surrogates; pwp chain for Xiji/Qingyang.
  * WTC: Morlet omega0=6; 300 AR(1) surrogate pairs; band summaries
    (2-16, 16-64, 64-128 d).
  * HMM: 2 Gaussian states on [d_tilt_z, sqrt(rain)_z]; 20 random restarts.
  * Event table: v2 definition (consecutive days >= 1 mm), full record.
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import ccm
import transfer_entropy as temod
import wavelet_coherence as wtc
import hmm_regimes as hm
import ml_response as ml
from common import load_panels, event_table, SITES, LO, FIG, CB, make_style, zscore, panel_letters

make_style()
T0 = time.time()


def jfloat(x):
    if x is None:
        return None
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else None
    except (TypeError, ValueError):
        return None


def longest_finite_segment(mask):
    """(start, end) of the longest run of True in a boolean mask."""
    best = (0, len(mask) - 1, 0)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i > best[2]:
                best = (i, j - 1, j - i)
            i = j
        else:
            i += 1
    return best[0], best[1]


out = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "design": "pre-registered choices in module docstring"},
       "validation": {}, "daily_ci": {}, "ccm": {}, "te": {}, "wtc": {},
       "hmm": {}, "event_perm": {}, "ml": {}}
aux = {}  # numpy arrays for figures

# ================================================================ 1. synthetic validations
print("[1/6] synthetic validations ...", flush=True)
v_ccm = ccm.synthetic_validation(n=3000, c_xy=0.02, c_yx=0.0)
out["validation"]["ccm"] = {
    "design": "unidirectionally coupled logistic maps (y drives x, c=0.02)",
    "true_direction_y_from_x": [[L, jfloat(m), jfloat(s)] for L, m, s in v_ccm["y_from_x"]],
    "reverse_x_from_y": [[L, jfloat(m), jfloat(s)] for L, m, s in v_ccm["x_from_y"]],
    "pass": bool(v_ccm["y_from_x"][-1][1] > 0.9 and
                 max(abs(m) for _, m, _ in v_ccm["x_from_y"]) < 0.1)}
out["validation"]["te"] = temod.synthetic_validation()
out["validation"]["wtc"] = wtc.synthetic_validation()
out["validation"]["hmm"] = hm.synthetic_validation()
print("   ccm pass:", out["validation"]["ccm"]["pass"],
      "| wtc:", {k: round(v, 3) for k, v in out["validation"]["wtc"].items()},
      "| hmm:", {k: (round(v, 3) if isinstance(v, float) else None)
                 for k, v in out["validation"]["hmm"].items()}, flush=True)

# ================================================================ 2. real data
panels = load_panels()
EVENTS = {s: event_table(panels[s]) for s in SITES}
out["n_events"] = {s: int(len(EVENTS[s])) for s in SITES}
print("[2/6] per-site causal / time-frequency / regime analyses ...", flush=True)

for s in SITES:
    d = panels[s]
    seg_mask = d["rain"].notna().values & d["tilt_mag"].notna().values
    i0, i1 = longest_finite_segment(seg_mask)
    seg = d.iloc[i0:i1 + 1]
    out["meta"][f"segment_{s}"] = {
        "start": str(seg.index[0].date()), "end": str(seg.index[-1].date()),
        "n_days": int(len(seg))}
    rain = seg["rain"].values.astype(float)
    tilt = seg["tilt_mag"].values.astype(float)
    dtilt = np.diff(tilt, prepend=tilt[0])[1:]
    rain_c, dtilt_c = rain[1:], dtilt  # aligned lengths for CCM/TE
    wet = (rain_c > 0).astype(int)
    xbin = (dtilt_c > np.median(dtilt_c)).astype(int)

    # ---- daily r (full record, v2-comparable) + segment block-bootstrap CI
    r_all = d["rain"].values.astype(float)
    t_all = d["tilt_mag"].values.astype(float)
    ok = np.isfinite(r_all) & np.isfinite(t_all)
    r_daily = float(np.corrcoef(r_all[ok], t_all[ok])[0, 1])
    rng = np.random.default_rng(42)
    r_seg = seg["rain"].values.astype(float)
    t_seg = seg["tilt_mag"].values.astype(float)
    n = len(r_seg)
    bl, reps = 30, 1000
    bs = []
    nb = int(np.ceil(n / bl))
    for _ in range(reps):
        idx = np.concatenate([rng.integers(0, n - bl) + np.arange(bl)
                              for _ in range(nb)])[:n]
        if np.std(r_seg[idx]) < 1e-12 or np.std(t_seg[idx]) < 1e-12:
            continue
        bs.append(np.corrcoef(r_seg[idx], t_seg[idx])[0, 1])
    out["daily_ci"][s] = {"r_full_record": r_daily,
                          "r_segment": float(np.corrcoef(r_seg, t_seg)[0, 1]),
                          "ci95": [jfloat(np.percentile(bs, 2.5)),
                                   jfloat(np.percentile(bs, 97.5))]}

    # ---- CCM: cross-map rain from the d-tilt manifold (primary) + reverse
    sig_fwd = ccm.ccm_significance(dtilt_c, rain_c, E=3, lags=range(0, 31),
                                   n_surr=200, seed=2)
    sig_rev = ccm.ccm_significance(rain_c, dtilt_c, E=3, lags=range(0, 31),
                                   n_surr=200, seed=3)
    lagged = ccm.lagged_ccm(dtilt_c, rain_c, E=3, lags=range(0, 31))
    # skill curve at the site's best lag (matches the reported best-lag skill)
    best_lag = sig_fwd["best_lag"]
    rain_shift = (np.concatenate([np.full(best_lag, np.nan), rain_c[:-best_lag]])
                  if best_lag > 0 else rain_c.copy())
    curve = ccm.ccm_skill_curve(dtilt_c, rain_shift, E=3, tau=1,
                                lib_fracs=(0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
                                n_boot=30, seed=1)
    out["ccm"][s] = {
        "skill_curve_rain_from_dtilt_at_best_lag": [[L, jfloat(m), jfloat(sd)] for L, m, sd in curve],
        "significance_rain_from_dtilt": {k: (jfloat(v) if isinstance(v, float) else v)
                                         for k, v in sig_fwd.items()},
        "significance_dtilt_from_rain": {k: (jfloat(v) if isinstance(v, float) else v)
                                         for k, v in sig_rev.items()},
        "lagged_skill_rain_from_dtilt": {int(k): jfloat(v) for k, v in lagged.items()}}
    aux[f"ccm_lag_{s}"] = np.array([lagged[k] for k in range(31)])
    print(f"   CCM {s}: fwd best lag {sig_fwd['best_lag']} skill {sig_fwd['skill']:.3f} "
          f"p={sig_fwd['p']:.3f} | rev skill {sig_rev['skill']:.3f} p={sig_rev['p']:.3f}",
          flush=True)

    # ---- transfer entropy (both directions) + pwp chain where available
    te_res = {"rain_to_dtilt": temod.te_profile(wet, xbin, lags=range(0, 11), n_surr=400, seed=10),
              "dtilt_to_rain": temod.te_profile(xbin, wet, lags=range(0, 11), n_surr=400, seed=20)}
    if "pwp" in seg.columns:
        pwpc = seg["pwp"].values.astype(float)[1:]
        pbin = (pwpc > np.nanmedian(pwpc)).astype(int)
        te_res["rain_to_pwp"] = temod.te_profile(wet, pbin, lags=range(0, 11), n_surr=400, seed=30)
        te_res["pwp_to_dtilt"] = temod.te_profile(pbin, xbin, lags=range(0, 11), n_surr=400, seed=40)
    out["te"][s] = {
        pair: {int(lag): {"te": jfloat(r["te"]), "z": jfloat(r["z"]), "p": jfloat(r["p"])}
               for lag, r in prof.items()}
        for pair, prof in te_res.items()}

    # ---- wavelet coherence (rain vs tilt) on the segment
    scales, w, p95 = wtc.wtc_significance(zscore(rain), zscore(tilt), n_surr=300,
                                          seed=5, s_min=2.0,
                                          s_max=min(len(seg) / (2 * 1.03), 128 / 1.03))
    mask_coi = wtc.coi_mask(len(seg), scales)
    periods = 1.03 * scales
    sigmap = (w > p95) & (~mask_coi)
    bands = {"2-16d": (periods >= 2) & (periods < 16),
             "16-64d": (periods >= 16) & (periods < 64),
             "64-128d": (periods >= 64) & (periods <= 128)}
    band_frac = {b: jfloat(sigmap[m].mean()) if m.any() else None
                 for b, m in bands.items()}
    area_ok = (~mask_coi).mean()
    out["wtc"][s] = {"periods": periods.tolist(),
                     "band_sig_fraction": band_frac,
                     "analysis_fraction": jfloat(area_ok)}
    aux[f"wtc_{s}"] = w.astype(np.float32)
    aux[f"wtc_p95_{s}"] = p95.astype(np.float32)
    aux[f"wtc_coi_{s}"] = mask_coi
    aux[f"wtc_periods_{s}"] = periods.astype(float)
    aux[f"wtc_time_{s}"] = seg.index.values.astype("datetime64[D]").astype(int)
    print(f"   WTC {s}: sig-frac {band_frac} (n={len(seg)} d)", flush=True)

    # ---- HMM regimes on the segment
    obs = np.column_stack([zscore(dtilt_c), zscore(np.sqrt(np.maximum(rain_c, 0.0)))])
    model = hm.fit_hmm(obs, K=2, n_init=20, seed=7)
    summ = hm.summarize(model)
    out["hmm"][s] = summ
    aux[f"hmm_path_{s}"] = np.array(summ["path_active"], dtype=np.int8)
    aux[f"hmm_time_{s}"] = seg.index.values[1:].astype("datetime64[D]").astype(int)
    print(f"   HMM {s}: A={[[round(v,3) for v in row] for row in summ['transition_matrix']]} "
          f"dwell={ [round(v,1) for v in summ['dwell_times']]} occ_active={summ['occupancy_active']:.3f}",
          flush=True)

    # ---- event-scale permutation test (v2-consistent event table)
    e = EVENTS[s].dropna(subset=["delta"])
    if len(e) > 5 and e["delta"].std() > 1e-12:
        from scipy.stats import spearmanr
        rho = float(spearmanr(e["rsum"], e["delta"]).statistic)
        rng = np.random.default_rng(11)
        null = [float(spearmanr(rng.permutation(e["rsum"].values), e["delta"].values).statistic)
                for _ in range(5000)]
        p_two = float(np.mean(np.abs(np.array(null)) >= abs(rho)))
    else:
        rho, p_two = None, None
    out["event_perm"][s] = {"n": int(len(e)), "spearman": jfloat(rho), "p_two_sided": jfloat(p_two),
                            "delta_var": jfloat(e["delta"].var())}

# ================================================================ 3. ML layer
print("[3/6] interpretable ML layer ...", flush=True)
ml_res, evs, pooled = ml.run_all()
att = ml.attribution(pooled)
out["ml"] = {**ml_res, "attribution": att}

# ================================================================ 4. figures
print("[4/6] figures ...", flush=True)
import matplotlib.dates as mdates

# ---- FIG 4: wavelet coherence
SC = {"Xiji": CB["blue"], "Qingyang": CB["red"], "Tianshui": CB["green"]}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
for ax, s in zip(axes, SITES):
    t = aux[f"wtc_time_{s}"]
    w = aux[f"wtc_{s}"]
    p95 = aux[f"wtc_p95_{s}"]
    coi = aux[f"wtc_coi_{s}"]
    periods = aux[f"wtc_periods_{s}"]
    im = ax.pcolormesh(mdates.num2date(t), periods, w, cmap="Blues",
                       vmin=0, vmax=0.9, shading="auto", rasterized=True)
    ax.contour(mdates.num2date(t), periods, (w > p95).astype(float) - 0.5,
               levels=[0], colors=[CB["red"]], linewidths=0.9)
    ax.contourf(mdates.num2date(t), periods, coi.astype(float), levels=[0.5, 1.5],
                colors=["white"], alpha=0.7, hatches=["///"])
    ax.set_yscale("log")
    ax.set_ylim(2, 128)
    ax.invert_yaxis()
    ax.set_title(f"{s} (n={out['meta'][f'segment_{s}']['n_days']} d)", fontsize=9)
    if s == "Xiji":
        ax.set_ylabel("period (d)")
    ax.tick_params(axis="x", rotation=30, labelsize=6)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
panel_letters(axes, x=-0.06, y=1.08)
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="wavelet coherence")
fig.savefig(os.path.join(FIG, "loess_adv_wtc.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "loess_adv_wtc.png"), bbox_inches="tight")
plt.close(fig)

# ---- FIG 5: CCM + TE
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for s in SITES:
    cur = out["ccm"][s]["skill_curve_rain_from_dtilt_at_best_lag"]
    L = [c[0] for c in cur]
    m = [c[1] for c in cur]
    sd = [c[2] for c in cur]
    axes[0].errorbar(np.array(L) / max(L), m, yerr=sd, color=SC[s],
                     marker="o", ms=3, lw=1.2, capsize=2, label=s)
    p95v = out["ccm"][s]["significance_rain_from_dtilt"]["surr_p95"]
    axes[0].axhline(p95v, color=SC[s], ls=":", lw=0.6, alpha=0.6)
axes[0].set_xlabel("library size (fraction of record)")
axes[0].set_ylabel("CCM skill  r(rain, mapped rain)")
axes[0].set_title("Rain cross-mapped from tilt dynamics\n(skill at each site's best lag)", fontsize=9)
axes[0].legend(fontsize=7, frameon=False)
lags = np.arange(11)
for s in SITES:
    z_fwd = [out["te"][s]["rain_to_dtilt"][int(l)]["z"] for l in lags]
    z_rev = [out["te"][s]["dtilt_to_rain"][int(l)]["z"] for l in lags]
    axes[1].plot(lags, z_fwd, color=SC[s], lw=1.2,
                 label=f"{s} rain" + r"$\to$" + "tilt")
    axes[1].plot(lags, z_rev, color=SC[s], lw=1.0, ls="--", alpha=0.7)
axes[1].axhspan(-2, 2, color="0.9", zorder=0)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_xlabel("lag (d)")
axes[1].set_ylabel("transfer entropy z")
axes[1].set_title("Directed information flow (solid rain" + r"$\to$" + "tilt)", fontsize=9)
axes[1].legend(fontsize=6, frameon=False)
panel_letters(axes, x=-0.12, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "loess_adv_ccm_te.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "loess_adv_ccm_te.png"), bbox_inches="tight")
plt.close(fig)

# ---- FIG 6: HMM regimes
fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.8), sharex=False)
for ax, s in zip(axes, SITES):
    tt = aux[f"hmm_time_{s}"]
    dates = mdates.num2date(tt)
    path = aux[f"hmm_path_{s}"].astype(float)
    dd = panels[s]
    rain_seg = dd["rain"].values[1:len(tt) + 1]
    dtilt_seg = dd["tilt_mag"].diff().values[1:len(tt) + 1]
    ax.fill_between(dates, 0, rain_seg, step="mid", color="0.88", zorder=0,
                    label="rain")
    ax.fill_between(dates, 0, rain_seg, step="mid", where=path > 0,
                    color=CB["lightblue"], zorder=1,
                    label="rain during active state")
    ax2 = ax.twinx()
    ax2.plot(dates, dtilt_seg, color=CB["grey"], lw=0.5)
    ax2.set_ylabel(r"$\Delta$tilt", fontsize=7, color="0.35")
    ax2.tick_params(labelsize=6, colors="0.35")
    h = out["hmm"][s]
    ax.set_ylabel("rain (mm)", fontsize=7)
    ax.set_title(f"{s}: HMM active state (occupancy {h['occupancy_active']:.0%}, "
                 f"dwell {h['dwell_times'][h['active_state']]:.1f} d)", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
panel_letters(axes, x=-0.10, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "loess_adv_hmm.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "loess_adv_hmm.png"), bbox_inches="tight")
plt.close(fig)

# ---- FIG 7: ML attribution (2x2: AUC, importance, PDP sin_doy, PDP api7)
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
sites = [s for s in SITES if s in out["ml"]["loso"]]
aucs = [out["ml"]["loso"][s]["AUC"] for s in sites]
brs = [out["ml"]["loso"][s]["base_rate_test"] for s in sites]
xs = np.arange(len(sites))
axes[0, 0].bar(xs, aucs, color=[SC[s] for s in sites], alpha=0.85)
axes[0, 0].scatter(xs, brs, marker="D", color="k", s=14, zorder=3, label="base rate")
axes[0, 0].axhline(0.5, color="k", lw=0.6, ls=":")
for x, a in zip(xs, aucs):
    axes[0, 0].text(x, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
axes[0, 0].set_xticks(xs)
axes[0, 0].set_xticklabels(sites, fontsize=7)
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_ylabel("LOSO AUC (response occurrence)")
axes[0, 0].set_title("Cross-site prediction", fontsize=9)
axes[0, 0].legend(fontsize=6, frameon=False)
imp = [t for t in out["ml"]["attribution"]["permutation_importance_AUC"][:8]][::-1]
axes[0, 1].barh([t[0] for t in imp], [t[1] for t in imp], color=CB["blue"], alpha=0.85)
axes[0, 1].set_xlabel("permutation importance (AUC drop)")
axes[0, 1].set_title("What predicts response occurrence", fontsize=9)
axes[0, 1].tick_params(axis="y", labelsize=6)
pdp = out["ml"]["attribution"]["pdp"]
pdp_axes = {"sin_doy": axes[1, 0], "api7": axes[1, 1]}
for i, (f, ax) in enumerate(pdp_axes.items()):
    if f not in pdp:
        ax.set_visible(False)
        continue
    cur = pdp[f]
    ax.plot(cur["grid"], cur["pdp"], color=[CB["blue"], CB["red"]][i], lw=1.4)
    ax.set_xlabel(f)
    ax.set_ylabel("P(response | feature)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Partial dependence: {f}", fontsize=9)
panel_letters(axes, x=-0.14, y=1.05)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "loess_adv_ml.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "loess_adv_ml.png"), bbox_inches="tight")
plt.close(fig)

# ================================================================ 5. save
print("[5/6] saving ...", flush=True)
with open(os.path.join(LO, "loess_advanced.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
np.savez_compressed(os.path.join(LO, "advanced_aux.npz"), **aux)
print(f"DONE advanced layer in {time.time() - T0:.0f}s")
