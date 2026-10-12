"""Validation-results figure for the French out-of-sample test (paper1 v3).
(a) weak daily correlation replicates: |r(rain, response)| for 3 loess + 3 French sites
(b) TE profiles rain->velocity-burst for the three French sites (z vs lag)
(c) state-gate effect on false-alarm rate: baseline vs gated (3 FR + 1 loess pooled)
Paper conventions: Arial, bold bare panel letters, clash palette, 600 dpi, PDF+PNG.
"""
import json
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import transfer_entropy as temod  # noqa: E402

ROOT = BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.8})

CB = ["#0000A0", "#CF3D3E", "#FFB400", "#35B257", "#9b71aa", "#7EA5E0"]
GREY = "#777777"

# ---- data: loess daily correlations ----
loess_corr = {}
for s in ("Xiji", "Qingyang", "Tianshui"):
    d = pd.read_csv(f"{BASE}/results/daily_{s}.csv", parse_dates=["date"]).set_index("date")
    tm = np.sqrt(d["tiltx"] ** 2 + d["tilty"] ** 2)
    c1 = abs(float(d["rain"].corr(tm)))
    c2 = abs(float(d["rain"].corr(tm.diff())))
    loess_corr[s] = (c1, c2)

# ---- data: French sites (corr + TE profiles) ----
FR = {"Sechilienne": "sechilienne", "Viella": "viella", "Villerville": "villerville"}
fr_corr, te_prof = {}, {}
import glob as _glob
for name, key in FR.items():
    d = [x for x in _glob.glob(f"{ROOT}/data/FrenchLandslides/*")
         if os.path.basename(x).lower().replace("é", "e").endswith(key)][0]
    hm = pd.read_csv(glob.glob(f"{d}/hm_features_*.csv")[0], parse_dates=["Date"]).set_index("Date")
    vel = pd.read_csv(glob.glob(f"{d}/vel_*.csv")[0], parse_dates=["Date"]).set_index("Date")
    vcols = [c for c in vel.columns if "_vel" in c or "decomposed" in c]
    v = vel[vcols].mean(axis=1)
    df = pd.DataFrame({"v": v}).join(hm["R_1"].rename("rain"), how="inner").dropna()
    fr_corr[name] = (abs(float(df["v"].corr(df["rain"]))),
                     abs(float(df["v"].diff().corr(df["rain"]))))
    dv = df["v"].diff()
    wet = (df["rain"] >= 1.0).astype(int).values
    dvb = (dv > dv.quantile(0.75)).astype(int).values[1:]
    te_prof[name] = temod.te_profile(wet[1:], dvb, lags=range(0, 11), n_surr=200, seed=10)


# ---- data: gate effect (false-alarm rate baseline vs gated) ----
ewi = json.load(open(f"{BASE}/results/loess_ewi_backcalc.json"))
fpt = json.load(open(f"{BASE}/results/french_pipeline_test.json"))
gate_pairs = [
    ("Loess (pooled)", ewi["pooled"]["baseline"]["false_alarm_rate"], ewi["pooled"]["gated"]["false_alarm_rate"]),
    ("Séchilienne", fpt["Sechilienne"]["A_delta_pos"]["baseline"]["false_alarm_rate"],
     fpt["Sechilienne"]["A_delta_pos"]["gated"]["false_alarm_rate"]),
    ("Viella", fpt["Viella"]["A_delta_pos"]["baseline"]["false_alarm_rate"],
     fpt["Viella"]["A_delta_pos"]["gated"]["false_alarm_rate"]),
    ("Villerville", fpt["Villerville"]["A_delta_pos"]["baseline"]["false_alarm_rate"],
     fpt["Villerville"]["A_delta_pos"]["gated"]["false_alarm_rate"]),
]

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
labs = ["a", "b", "c"]
for a, L in zip(ax, labs):
    a.text(-0.02, 1.06, L, transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")

# (a) weak daily correlation replicates
names = list(loess_corr) + list(fr_corr)
vals1 = [loess_corr[s][0] for s in loess_corr] + [fr_corr[s][0] for s in fr_corr]
vals2 = [loess_corr[s][1] for s in loess_corr] + [fr_corr[s][1] for s in fr_corr]
x = np.arange(len(names))
ax[0].bar(x - 0.2, vals1, 0.38, color=CB[0], label="level")
ax[0].bar(x + 0.2, vals2, 0.38, color=CB[1], label="increment")
ax[0].axhline(0.23, color=GREY, ls="--", lw=0.8)
ax[0].text(len(names) - 0.4, 0.245, "loess max", fontsize=6, color=GREY, ha="right")
ax[0].set_xticks(x)
ax[0].set_xticklabels(["Xiji", "Qingyang", "Tianshui", "Séchil.", "Viella", "Villerv."],
                      rotation=35, ha="right", fontsize=6.5)
for i in range(3, 6):
    ax[0].get_xticklabels()[i].set_color(CB[3])
ax[0].set_ylabel("|r(rain, response)|")
ax[0].legend(fontsize=6, frameon=False, loc="upper right")
ax[0].set_title("Weak daily correlation replicates", fontsize=7.5, pad=16)

# (b) TE profiles
for i, (name, prof) in enumerate(te_prof.items()):
    lags = sorted(int(k) for k in prof)
    zs = [prof[l]["z"] for l in lags]
    ax[1].plot(lags, zs, "-o", color=CB[i], ms=3, lw=1.2, label=name)
ax[1].axhline(2, color=GREY, ls="--", lw=0.8)
ax[1].text(9.8, 2.4, "z=2", fontsize=6, color=GREY, ha="right")
ax[1].axvline(3, color=CB[0], ls=":", lw=0.7)
ax[1].set_xlabel("lag (days)")
ax[1].set_ylabel("TE z (rain → burst)")
ax[1].legend(fontsize=6, frameon=False)
ax[1].set_title("Delayed transfer replicates (or is power-limited)", fontsize=7.5, pad=16)

# (c) gate effect on false-alarm rate
x = np.arange(len(gate_pairs))
for i, (name, b, g) in enumerate(gate_pairs):
    ax[2].bar(i - 0.2, b, 0.38, color=CB[0])
    ax[2].bar(i + 0.2, g, 0.38, color=CB[1])
    ax[2].text(i + 0.2, g + 0.012, f"{g:.2f}", fontsize=6, ha="center", color=CB[1])
ax[2].set_xticks(x)
ax[2].set_xticklabels([n for n, _, _ in gate_pairs], rotation=25, ha="right", fontsize=6.5)
ax[2].set_ylabel("false-alarm rate")
custom = [plt.Rectangle((0, 0), 1, 1, color=CB[0]), plt.Rectangle((0, 0), 1, 1, color=CB[1])]
ax[2].legend(custom, ["trigger only", "trigger + state gate"], fontsize=6, frameon=False, loc="upper left")
ax[2].set_ylim(0, 0.75)
ax[2].set_title("State gate lowers false alarms", fontsize=7.5, pad=16)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(f"{BASE}/analysis/figures/fig_french_validation.pdf", dpi=600)
fig.savefig(f"{BASE}/analysis/figures/fig_french_validation.png", dpi=600)
print("saved fig_french_validation.pdf/png")
