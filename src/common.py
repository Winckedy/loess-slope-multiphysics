#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared data loading + palette for the advanced-analysis layer (paper2 v3).
Reads only results/daily_*.csv (derived from the provided monitoring set).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LO = os.path.join(ROOT, "results")
ADV = os.path.join(ROOT, "analysis", "advanced")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# Project palette (fixed): use in order; grey = neutral
PAL = ["#0000A0", "#CF3D3E", "#FFB400", "#FBDD85", "#35B257", "#7EA5E0"]
CB = {"blue": PAL[0], "red": PAL[1], "yellow": PAL[2], "cream": PAL[3],
      "green": PAL[4], "lightblue": PAL[5], "grey": "#8C8C8C"}

SITES = ["Xiji", "Qingyang", "Tianshui"]
SITE_COLOR = {"Xiji": CB["blue"], "Qingyang": CB["red"], "Tianshui": CB["green"]}


def load_panels():
    """Load daily panels, add tilt_mag and its first difference."""
    panels = {}
    for s in SITES:
        df = pd.read_csv(os.path.join(LO, f"daily_{s}.csv"), parse_dates=["date"]).set_index("date")
        df = df.sort_index()
        df["tilt_mag"] = np.sqrt(df.tiltx ** 2 + df.tilty ** 2)
        df["d_tilt"] = df["tilt_mag"].diff()
        panels[s] = df
    return panels


def zscore(x):
    x = np.asarray(x, dtype=float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd if sd > 1e-12 else 1.0)


def panel_letters(axs, x=-0.10, y=1.04, size=9, letters=None):
    """Nature-style panel labels: bold lowercase letters, top-left, NO parentheses.

    User-fixed 2026-09-29: in-figure labels use bare letters (a, b, c); the
    parenthesised form (a)(b)(c) lives in the LaTeX caption only.
    Skips invisible axes (e.g. hidden PDP panels).
    """
    try:
        flat = [a for a in np.ravel([axs]) if a is not None]
    except (TypeError, ValueError):
        flat = [axs]
    flat = [a for a in flat if a.get_visible()]
    if letters is None:
        letters = [chr(97 + i) for i in range(len(flat))]
    for a, L in zip(flat, letters):
        a.text(x, y, L, transform=a.transAxes, fontsize=size,
               fontweight="bold", va="bottom", ha="right")


def event_table(d, thresh=1.0):
    """Rainfall events = maximal runs of CONSECUTIVE days with rain>=thresh
    (identical to the v2 definition, for comparability). Post-event response =
    peak tilt magnitude in the 3 days after the event minus the 7-day pre-event
    baseline (nan-robust against logger gaps). Returns a DataFrame with columns:
      start,end,rsum,dur,imax,iint,api7,api30,tilt_base,delta,plus,doy.
    """
    rain = d["rain"].values
    tilt = d["tilt_mag"].values
    n = len(rain)
    wet = rain >= thresh
    ev_id = np.zeros(n, dtype=int)
    cur, in_ev = 0, False
    for i in range(n):
        if wet[i]:
            if not in_ev:
                cur += 1
                in_ev = True
            ev_id[i] = cur
        else:
            in_ev = False
    api = d["rain"].rolling(7, min_periods=1).sum().shift(1).fillna(0.0).values
    api30 = d["rain"].rolling(30, min_periods=1).sum().shift(1).fillna(0.0).values
    rows = []
    for ev in range(1, cur + 1):
        idx = np.where(ev_id == ev)[0]
        start, end = idx[0], idx[-1]
        base = np.nanmean(tilt[max(0, start - 7):start]) if start > 0 else tilt[start]
        post = tilt[end + 1:end + 4]
        if len(post) == 0 or not np.isfinite(base):
            continue
        delta = np.nanmax(post) - base
        if not np.isfinite(delta):
            continue
        rows.append(dict(start=start, end=end,
                         rsum=rain[idx].sum(), dur=len(idx), imax=rain[idx].max(),
                         iint=rain[idx].sum() / len(idx),
                         api7=api[start], api30=api30[start],
                         tilt_base=base, delta=delta, plus=float(delta > 0),
                         doy=int(d.index[start].dayofyear)))
    return pd.DataFrame(rows)


def make_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "figure.dpi": 300, "savefig.dpi": 300})
