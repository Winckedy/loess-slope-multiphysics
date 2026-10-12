"""Early-warning back-calculation: does the state gate (season + API7) improve
a classical intensity-duration trigger?

Design (pre-registered here, no peeking):
  Events    = paper's v2 catalogue (1 mm consecutive days; response = peak tilt
              increment in 3 post-event days above 7-day pre-event baseline).
  Baseline  = logistic trigger surface on (I, D) [identical to M3], alert when
              P(response) >= 0.5.
  Gated     = alert only when the state gate is OPEN AND baseline fires.
  Gate      = API7 >= site median (50th pct of all events) AND event start in
              the site's wet season (months whose climatological rainfall
              exceeds the monthly mean), both fixed from the full record.
  Metrics   = n alerts, precision, recall, false-alarm rate FP/(FP+TN),
              miss rate FN/(TP+FN). Reported pooled and per site.
"""
import os
import json, os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import common as C  # noqa: E402

RNG = np.random.default_rng(42)


def month_wet_mask(d):
    """Months with climatological rainfall above the monthly mean = wet season."""
    mon = d.groupby(d.index.month)["rain"].mean()
    return mon >= mon.mean()


def metrics(y_true, y_alert):
    tp = int(((y_true == 1) & (y_alert == 1)).sum())
    fp = int(((y_true == 0) & (y_alert == 1)).sum())
    tn = int(((y_true == 0) & (y_alert == 0)).sum())
    fn = int(((y_true == 1) & (y_alert == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    miss = fn / (tp + fn) if tp + fn else np.nan
    return dict(alerts=tp + fp, tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(prec, 3), recall=round(rec, 3),
                false_alarm_rate=round(fpr, 3), miss_rate=round(miss, 3))


panels = C.load_panels()
wet_months = {s: month_wet_mask(panels[s]) for s in C.SITES}

rows_all = []
per_site = {}
for s in C.SITES:
    ev = C.event_table(panels[s]).dropna(subset=["tilt_base", "delta"])
    ev = ev[(ev["iint"] > 0) & (ev["dur"] > 0)].reset_index(drop=True)
    if len(ev) < 10:
        continue
    # gate: API7 >= site event median AND wet-season start
    api_med = float(ev["api7"].median())
    doy = ev["doy"].values
    month = ((doy - 1) // 30 + 1).clip(1, 12)  # coarse month from doy
    gate_open = ((ev["api7"].values >= api_med)
                 & np.isin(month, np.where(wet_months[s].values)[0] + 1)).astype(int)
    X = np.c_[ev["iint"].values, ev["dur"].values, ev["iint"].values * ev["dur"].values]
    y = ev["plus"].values.astype(int)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    base_alert = (p >= 0.5).astype(int)
    gated_alert = ((p >= 0.5) & (gate_open == 1)).astype(int)
    per_site[s] = {
        "n_events": int(len(ev)), "api7_median": round(api_med, 1),
        "n_gate_open": int(gate_open.sum()),
        "baseline": metrics(y, base_alert),
        "gated": metrics(y, gated_alert),
    }

# pooled across sites (refit on all events, site-blind)
pool = []
for s in C.SITES:
    ev = C.event_table(panels[s]).dropna(subset=["tilt_base", "delta"])
    ev = ev[(ev["iint"] > 0) & (ev["dur"] > 0)]
    api_med = float(ev["api7"].median())
    month = ((ev["doy"].values - 1) // 30 + 1).clip(1, 12)
    ev = ev.assign(gate=((ev["api7"].values >= api_med)
                         & np.isin(month, np.where(wet_months[s].values)[0] + 1)).astype(int))
    pool.append(ev)
pool = pd.concat(pool, ignore_index=True)
Xp = np.c_[pool["iint"], pool["dur"], pool["iint"] * pool["dur"]]
yp = pool["plus"].values.astype(int)
pipe = Pipeline([("sc", StandardScaler()),
                 ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
pipe.fit(Xp, yp)
pp = pipe.predict_proba(Xp)[:, 1]
pooled = {
    "n_events": int(len(pool)),
    "baseline": metrics(yp, (pp >= 0.5).astype(int)),
    "gated": metrics(yp, ((pp >= 0.5) & (pool["gate"].values == 1)).astype(int)),
}

out = {"design": "alert = P(response | I,D) >= 0.5; gated adds API7>=site-median AND wet-season",
       "per_site": per_site, "pooled": pooled}
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "loess_ewi_backcalc.json")
json.dump(out, open(path, "w"), indent=1)
print(json.dumps(out, indent=1))
print("saved:", path)
