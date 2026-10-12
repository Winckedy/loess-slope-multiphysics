#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Additional display figures for paper2 v3 (user request: more figures, less prose).

Produces six new PDFs (+PNG previews) in analysis/figures:
  fig_coverage.pdf       - station/device data-coverage Gantt with gaps
  fig_wetfront.pdf       - depth-time moisture heatmap (Qingyang 10-depth profile) + rain
  fig_overview.pdf       - 3-site multi-physics daily overview (rain / pwp / tilt)
  fig_event_response.pdf - event-scale scatter + binned response gate (regenerated,
                           canonical palette; replaces legacy loess_event.pdf)
  fig_lagprofiles.pdf    - CCM lagged skill + TE lag-profile chains (Xiji, Qingyang)
  fig_m3_forest.pdf      - M3 intensity-duration coefficient forest plot (alpha, beta CI95)

Style: common.make_style() (Arial 9 pt, despined, 300 dpi) + fixed six-colour palette.
Site colours follow common.SITE_COLOR (Xiji blue, Qingyang red, Tianshui green).
"""
import os, json, glob, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import (ROOT, LO, FIG, CB, SITE_COLOR, SITES,
                             load_panels, event_table, make_style, panel_letters)

make_style()

# ------------------------------------------------------------------ helpers
def read_csv_cn(path):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise IOError(path)

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

profil = json.load(open(os.path.join(ROOT, "results", "raw_record_profile.json"),
                        encoding="utf-8"))

# device-level gap positions: read raw logger date columns for devices with >=1 gap day
def device_gap_dates(pattern, tcol="上报时间"):
    ds = set()
    for p in sorted(glob.glob(pattern)):
        df = read_csv_cn(p)
        t = pd.to_datetime(df[tcol], errors="coerce").dropna().dt.normalize()
        ds.update(t.unique())
    ds = pd.DatetimeIndex(sorted(ds))
    if len(ds) < 2:
        return pd.DatetimeIndex([])
    return pd.date_range(ds.min(), ds.max(), freq="D").difference(ds)

RAW = os.path.join(ROOT, "data")
_gap_specs = {
    "QY_slide": os.path.join(RAW, "Qingyang", "滑坡", "*.csv"),
    "TS_met": os.path.join(RAW, "Tianshui", "21609*.csv"),
    "TS_level": os.path.join(RAW, "Tianshui", "液位", "*.csv"),
}
raw_gaps = {}
for key, pat in _gap_specs.items():
    p = profil[key]
    if p.get("gap_days_total", 0) >= 1:
        raw_gaps[key] = device_gap_dates(pat)
        print(f"gap {key}: {len(raw_gaps[key])} missing days "
              f"({', '.join(str(d.date()) for d in raw_gaps[key][:1])} ...)")
adv = json.load(open(os.path.join(LO, "loess_advanced.json"), encoding="utf-8"))
nov = json.load(open(os.path.join(LO, "loess_novel.json"), encoding="utf-8"))
six = json.load(open(os.path.join(LO, "loess_six_methods.json"), encoding="utf-8"))
panels = load_panels()

# ================================================================== 1. coverage
def missing_days(df):
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return full.difference(df.index)

daily_gaps = {s: missing_days(panels[s]) for s in SITES}
rows = []  # (label, site, t_min, t_max, gap_days, gap_dates)
xj = [k for k in profil if k.startswith("Xiji")]
for k in sorted(profil):
    if k in ("QY_met", "QY_soil", "QY_slide", "TS_met", "TS_level", "TS_slide") or k.startswith("Xiji"):
        p = profil[k]
        t0, t1 = pd.Timestamp(p["t_min"]), pd.Timestamp(p["t_max"])
        lab_map = {
            "Xiji_slide": ("Xiji  slide logger (5-min)", "Xiji"),
            "Xiji_soil":  ("Xiji  soil profile (10-min)", "Xiji"),
            "Xiji_met":   ("Xiji  weather station", "Xiji"),
            "QY_met": ("Qingyang  weather station (10-min)", "Qingyang"),
            "QY_soil": ("Qingyang  soil profile, 10 depths (10-min)", "Qingyang"),
            "QY_slide": ("Qingyang  slide logger, pwp + tilt (5-min)", "Qingyang"),
            "TS_met": ("Tianshui  weather station (5-min)", "Tianshui"),
            "TS_level": ("Tianshui  laser water level (60-min)", "Tianshui"),
            "TS_slide": ("Tianshui  slide logger (5-min)", "Tianshui"),
        }
        lab, site = lab_map.get(k, (k, "Qingyang"))
        gap_dates, gapd = [], p.get("gap_days_total", 0.0)
        if k in raw_gaps:
            gap_dates = raw_gaps[k]
        rows.append((lab, site, t0, t1, gapd, gap_dates))

order = {"Xiji": 0, "Qingyang": 1, "Tianshui": 2}
rows.sort(key=lambda r: (order[r[1]], r[2]))
if not any(r[1] == "Xiji" for r in rows):  # Xiji raw data is one xlsx; use daily panel
    dx = panels["Xiji"]
    rows.append(("Xiji  slide logger (daily panel)", "Xiji",
                 dx.index.min(), dx.index.max(), float(len(daily_gaps["Xiji"])),
                 daily_gaps["Xiji"]))
    rows.sort(key=lambda r: (order[r[1]], r[2]))

fig, ax = plt.subplots(figsize=(6.3, 2.6))
for i, (lab, site, t0, t1, gapd, gap_dates) in enumerate(rows):
    y = len(rows) - 1 - i
    c = SITE_COLOR[site]
    ax.barh(y, (t1 - t0).days, left=mdates.date2num(t0), height=0.62,
            color=c, alpha=0.82, edgecolor="none")
    for g in gap_dates:
        ax.barh(y, 1.0, left=mdates.date2num(g), height=0.62,
                color="white", edgecolor=c, lw=0.4)
    # merged gap ranges as hatched blocks (visible at multi-year scale)
    if len(gap_dates):
        gs = pd.Series(sorted(gap_dates))
        brk = gs.diff() > pd.Timedelta(days=1)
        grp = brk.cumsum()
        for _, chunk in gs.groupby(grp):
            g0, g1 = chunk.iloc[0], chunk.iloc[-1]
            ax.barh(y, (g1 - g0).days + 1.0, left=mdates.date2num(g0), height=0.62,
                    facecolor="white", edgecolor=c, lw=0.5, hatch="////", zorder=5)
    ax.text(mdates.date2num(t1) + 6, y, f"{gapd:.0f} d gap" if gapd >= 1 else "no gap",
            va="center", fontsize=6.5, color="#555555")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=7.5)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
ax.set_xlim(pd.Timestamp("2023-06-15"), pd.Timestamp("2026-01-31"))
ax.set_xlabel("2023 – 2025")
legend = [Patch(fc=SITE_COLOR[s], alpha=0.82, label=s) for s in SITES]
legend.append(Patch(fc="white", ec="#888888", label="missing day"))
ax.legend(handles=legend, fontsize=6.5, frameon=False, ncol=4,
          loc="upper center", bbox_to_anchor=(0.5, 1.14))
save(fig, "fig_coverage")

# ================================================================== 2. wetting front
qy_dirs = glob.glob(os.path.join(ROOT, "data", "Qingyang", "水体", "*.csv"))
DEPTHS = [10, 20, 40, 60, 80, 110, 120, 140, 180, 200]
frames = []
for p in sorted(qy_dirs):
    df = read_csv_cn(p)
    df["上报时间"] = pd.to_datetime(df["上报时间"], errors="coerce")
    frames.append(df.set_index("上报时间"))
soil = pd.concat(frames).sort_index()
moist = pd.DataFrame(index=soil.index)
for d in DEPTHS:
    moist[d] = pd.to_numeric(soil[f"水分含量(%)-{d}cm"], errors="coerce")
moist = moist.where((moist > 0.5) & (moist < 60))
md = moist.resample("D").mean()
rain_qy = panels["Qingyang"]["rain"]

n_zero = int((moist <= 0.5).sum().sum())
fig, ax = plt.subplots(2, 1, figsize=(6.3, 4.1), sharex=True,
                       gridspec_kw={"height_ratios": [1, 3.1], "hspace": 0.08})
ax[0].bar(rain_qy.index, rain_qy.values, width=1.4, color=CB["blue"], lw=0)
ax[0].set_ylabel("rain\n(mm d$^{-1}$)", fontsize=8)
ax[0].set_ylim(0, rain_qy.max() * 1.15)
X, Y = np.meshgrid(md.index.values, np.array(DEPTHS, dtype=float))
pc = ax[1].pcolormesh(X, Y, md.T.values, cmap="YlGnBu", shading="nearest",
                      vmin=np.nanpercentile(md.values, 2),
                      vmax=np.nanpercentile(md.values, 98))
ax[1].invert_yaxis()
ax[1].set_ylabel("sensor depth (cm)", fontsize=8)
ax[1].set_yticks([10, 40, 80, 120, 160, 200])
cb = fig.colorbar(pc, ax=ax[1], pad=0.015, aspect=18)
cb.set_label("soil water content (% w/w)", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
ax[1].xaxis.set_major_locator(mdates.YearLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax[1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
print(f"wetfront: {md.shape[0]} days x {len(DEPTHS)} depths, masked low/high cells={n_zero}, "
      f"nan frac={np.isnan(md.values).mean():.3f}")
save(fig, "fig_wetfront")

# ================================================================== 3. overview
fig, axes = plt.subplots(3, 3, figsize=(6.3, 4.4), sharex="col",
                         gridspec_kw={"hspace": 0.50, "wspace": 0.34})
for j, s in enumerate(SITES):
    d = panels[s]
    ax0, ax1, ax2 = axes[:, j]
    ax0.bar(d.index, d["rain"].values, width=1.6, color=CB["blue"], lw=0)
    ax0.set_ylim(0, 65)
    if "pwp" in d.columns and d["pwp"].notna().sum() > 50:
        ax1.plot(d.index, d["pwp"], color=CB["green"], lw=0.55)
        ax1.set_ylabel("pwp (kPa)", fontsize=7.5)
    else:  # Tianshui: near-surface moisture instead of pwp
        ax1.plot(d.index, d["moist_mean"], color=CB["green"], lw=0.55)
        ax1.text(0.03, 0.9, "moisture (no pwp channel)", transform=ax1.transAxes,
                 fontsize=6.2, color="#555555", va="top")
    ax2.plot(d.index, d["tilt_mag"], color=CB["red"], lw=0.4)
    ax2.set_ylabel("tilt (deg)", fontsize=7.5)
    ax0.set_title(s, fontsize=9, color=SITE_COLOR[s], fontweight="bold")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0, 0].set_ylabel("rain\n(mm d$^{-1}$)", fontsize=7.5)
axes[2, 1].set_xlabel("record window per site (daily resampled)", fontsize=8)
panel_letters(axes, x=-0.13, y=1.07)
save(fig, "fig_overview")

# ================================================================== 4. event response (regenerated)
ev_all = []
for s in SITES:
    e = event_table(panels[s]).copy()
    e["site"] = s
    ev_all.append(e)
ev = pd.concat(ev_all, ignore_index=True)
bins = ["<5", "5-15", "15-30", ">30"]
binfo = {s: nov["event_scale"][s].get("binned", {}) for s in SITES}

fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
for s in SITES:
    sub = ev[ev.site == s]
    ax[0].scatter(sub.rsum, sub.delta, s=11, alpha=0.55, lw=0,
                  color=SITE_COLOR[s], label=f"{s} (n={len(sub)})")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xlabel("event rainfall sum (mm)")
ax[0].set_ylabel("post-event peak tilt change (deg)")
ax[0].set_ylim(-0.12, 0.32)
ax[0].legend(fontsize=6.5, frameon=False, loc="upper right")
ax[0].text(0.02, 0.04, "a", transform=ax[0].transAxes, fontweight="bold", fontsize=9)

xpos = np.arange(len(bins))
for s in SITES:
    ppos = [binfo[s].get(b, {}).get("p_pos", np.nan) for b in bins]
    ax[1].plot(xpos, ppos, marker="o", ms=4, lw=1.4, color=SITE_COLOR[s], label=s)
    si = SITES.index(s)
    for xi, b in enumerate(bins):
        n = binfo[s].get(b, {}).get("n", 0)
        if n and np.isfinite(ppos[xi]):
            off = (0, -11) if xi > 0 else (14, -4 + 11 * (si - 1))
            ax[1].annotate(f"n={n}", (xi, ppos[xi]), textcoords="offset points",
                           xytext=off, ha="center", fontsize=5.6, color="#555555")
ax[1].axhline(0.5, color="#888888", lw=0.6, ls=":")
ax[1].set_xticks(xpos); ax[1].set_xticklabels(bins)
ax[1].set_xlabel("event rainfall sum (mm)")
ax[1].set_ylabel("P(tilt response > 0)")
ax[1].set_ylim(0.28, 1.08)
ax[1].legend(fontsize=6.5, frameon=False, loc="lower left")
ax[1].text(0.02, 0.94, "b", transform=ax[1].transAxes, fontweight="bold", fontsize=9)
save(fig, "fig_event_response")

# ================================================================== 5. lag profiles
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3),
                        gridspec_kw={"wspace": 0.34})
axA, axB, axC = axs
for s in SITES:
    ccm = adv["ccm"][s]
    lag = {int(k): v for k, v in ccm["lagged_skill_rain_from_dtilt"].items()}
    lg = sorted(lag)
    axA.plot(lg, [lag[k] for k in lg], color=SITE_COLOR[s], lw=1.2,
             marker="o", ms=2.5, label=s)
    bl = int(ccm["significance_rain_from_dtilt"]["best_lag"])
    p95 = ccm["significance_rain_from_dtilt"]["surr_p95"]
    axA.axhline(p95, color=SITE_COLOR[s], lw=0.6, ls=":", alpha=0.8)
    axA.plot([bl], [lag[bl]], marker="*", ms=8, color=SITE_COLOR[s], ls="none")
axA.set_xlabel("lag rain leads tilt (d)")
axA.set_ylabel("cross-map skill (ρ)")
axA.set_ylim(-0.03, 0.44)
axA.legend(fontsize=6, frameon=False, loc="upper right")
axA.text(0.03, 0.95, "a", transform=axA.transAxes, fontweight="bold", fontsize=9, va="top")

def te_panel(axx, s, letter):
    te = adv["te"][s]
    def prof(key):
        p = {int(k): v["z"] for k, v in te[key].items()}
        ks = sorted(p)
        return ks, [p[k] for k in ks]
    ks, zs = prof("rain_to_dtilt")
    axx.plot(ks, zs, color=SITE_COLOR[s], lw=1.2, label="rain→Δtilt")
    ks, zs = prof("dtilt_to_rain")
    axx.plot(ks, zs, color=SITE_COLOR[s], lw=1.0, ls="--", label="Δtilt→rain")
    if "rain_to_pwp" in te:
        ks, zs = prof("rain_to_pwp")
        axx.plot(ks, zs, color=CB["green"], lw=1.0, label="rain→pwp")
    if "pwp_to_dtilt" in te:
        ks, zs = prof("pwp_to_dtilt")
        axx.plot(ks, zs, color=CB["red"], lw=1.0, label="pwp→Δtilt")
    axx.axhspan(-2, 2, color="#BBBBBB", alpha=0.35, lw=0)
    axx.axhline(0, color="k", lw=0.5)
    axx.set_xlabel("lag (d)")
    axx.set_xticks([0, 2, 4, 6, 8, 10])
    axx.text(0.04, 0.94, letter, transform=axx.transAxes, fontweight="bold", fontsize=9, va="top")

te_panel(axB, "Xiji", "b")
te_panel(axC, "Qingyang", "c")
axB.set_ylabel("transfer-entropy z")
axB.legend(fontsize=5.4, frameon=False, loc="upper right")
axC.legend(fontsize=5.4, frameon=False, loc="upper right")
save(fig, "fig_lagprofiles")

# ================================================================== 6. M3 forest
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0), sharey=True,
                       gridspec_kw={"wspace": 0.06})
site_names = {"Xiji": "Xiji", "Qingyang": "Qingyang", "Tianshui": "Tianshui"}
for panel, key, cikey, xlab in ((0, "alpha", "alpha_ci95",
                                 "intercept $\\alpha$ (log–log)"),
                                (1, "beta", "beta_ci95",
                                 "duration exponent $\\beta$")):
    for i, s in enumerate(SITES):
        m3 = six["m3"][s]
        val, (lo_, hi_) = m3[key], m3[cikey]
        ax[panel].errorbar(val, 2 - i, xerr=[[val - lo_], [hi_ - val]],
                           fmt="o", ms=4.5, lw=1.3, color=SITE_COLOR[s],
                           elinewidth=1.2, capsize=2.5)
    ax[panel].axvline(0, color="k", lw=0.7, ls="--")
    ax[panel].set_xlabel(xlab, fontsize=8)
ax[0].set_yticks([2, 1, 0]); ax[0].set_yticklabels([site_names[s] for s in SITES])
ax[1].text(0.04, 0.06, "$\\beta$ CI spans zero at Xiji and Tianshui;\nnegative at Qingyang",
           transform=ax[1].transAxes, fontsize=6.3, color="#555555")
# panel letters INSIDE each panel's upper-left corner (outside placement made
# "b" land above panel a); same in-panel style as fig_lagprofiles
for _ax, _L in zip(ax, "ab"):
    _ax.text(0.03, 0.97, _L, transform=_ax.transAxes, fontweight="bold",
             fontsize=9, va="top", ha="left", zorder=5)
save(fig, "fig_m3_forest")

print("ALL FIGURES DONE")
