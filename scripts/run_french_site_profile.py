"""Profile the French landslide datasets for pipeline adaptation (Option A)."""
import glob
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "FrenchLandslides")
_dirmap = {}
for d in glob.glob(f"{BASE}/*"):
    if os.path.isdir(d):
        _dirmap[os.path.basename(d).lower().replace("é", "e")] = d
SITES = {"Sechilienne": "sechilienne", "Viella": "viella", "Villerville": "villerville"}

out = {}
for site, dirname in SITES.items():
    d = _dirmap[dirname.lower()]
    vel_f = glob.glob(f"{d}/vel_*.csv")[0]
    hm_f = glob.glob(f"{d}/hm_features_*.csv")[0]
    vel = pd.read_csv(vel_f, parse_dates=["Date"]).set_index("Date")
    hm = pd.read_csv(hm_f, parse_dates=["Date"]).set_index("Date")
    rain = hm["R_1"] if "R_1" in hm.columns else None
    vcols = [c for c in vel.columns if c.endswith("_vel_processed") or "_decomposed" in c]
    v = vel[vcols].mean(axis=1)  # representative velocity: mean across plots

    # event count estimate with the paper's definition (1 mm consecutive days)
    r = rain.dropna()
    wet = (r >= 1.0).values
    n_ev, in_ev = 0, False
    for w in wet:
        if w and not in_ev:
            n_ev += 1
            in_ev = True
        elif not w:
            in_ev = False
    # daily correlation rain vs velocity (raw, like the paper's baseline)
    vv = v.reindex(r.index)
    dv = vv.diff()
    corr_v = vv.corr(r)
    corr_dv = dv.corr(r)

    out[site] = dict(
        vel_file=os.path.basename(vel_f), hm_file=os.path.basename(hm_f),
        n_rows_vel=len(vel), n_rows_hm=len(hm),
        t_min=str(vel.index.min().date()), t_max=str(vel.index.max().date()),
        n_vel_plots=len(vcols), vel_plots=[c.split("_")[0] for c in vcols],
        rain_missing=int(r.isna().sum()), rain_total_mm=round(float(r.sum()), 0),
        vel_missing_frac=round(float(v.isna().mean()), 3),
        vel_med_cm_d=round(float(v.median()), 4),
        n_events_1mm=n_ev,
        corr_rain_vel=round(float(corr_v), 3),
        corr_rain_dvel=round(float(corr_dv), 3),
    )
    p = out[site]
    print(f"--- {site} ---")
    print(f"  span {p['t_min']} .. {p['t_max']}  rows: vel={p['n_rows_vel']} hm={p['n_rows_hm']}")
    print(f"  plots({p['n_vel_plots']}): {p['vel_plots']}")
    print(f"  rain: missing={p['rain_missing']} d, total={p['rain_total_mm']} mm")
    print(f"  vel: missing={p['vel_missing_frac']:.1%}, median={p['vel_med_cm_d']} cm/d")
    print(f"  events(1mm consec): {n_ev}")
    print(f"  daily corr: rain~vel={p['corr_rain_vel']}  rain~dvel={p['corr_rain_dvel']}")

json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "french_sites_profile.json")
import json
json.dump(out, open(json_path, "w"), indent=1)
print("saved:", json_path)
