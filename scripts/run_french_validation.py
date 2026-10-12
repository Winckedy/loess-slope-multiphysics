"""Out-of-sample validation of the paper's pipeline on three French slow-moving
landslides (Sechilienne / Viella / Villerville; OMIV data, BEJEAN-MAILLARD et al. 2026).

Pre-registered design:
  Events    = 1 mm consecutive-day rain events (paper's v2 definition);
              delta = max velocity increment in 3 post-event days minus 7-day
              pre-event mean velocity. Response variants (both, as pre-agreed):
                A: delta > 0          (literal alignment with paper)
                B: delta > site q75   (velocity burst = response)
  Trigger   = logistic surface on (I, D, I*D), CV-AUC (5-fold stratified).
  Gate      = API7 >= site median AND wet-season start; metrics baseline vs gated.
  Ladder day-scale layer:
    CCM: rain cross-mapped FROM the mean-velocity manifold (E=3, lags 0-30).
    TE : wet-day indicator -> velocity-increment indicator, lags 0-10 (200 surr).
  Attribution: standardised logistic coefficients (I, D, API7, sin(doy), cos(doy)).
"""
import os
import json, os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import ccm  # noqa: E402
import transfer_entropy as temod  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "FrenchLandslides")
SITES = {"Sechilienne": "sechilienne", "Viella": "viella", "Villerville": "villerville"}
RNG = np.random.default_rng(42)


def load_site(key):
    import glob, os
    d = None
    for dd in glob.glob(f"{BASE}/*"):
        if os.path.basename(dd).lower().replace("é", "e") == key.lower():
            d = dd
    vel = pd.read_csv(glob.glob(f"{d}/vel_*.csv")[0], parse_dates=["Date"]).set_index("Date")
    hm = pd.read_csv(glob.glob(f"{d}/hm_features_*.csv")[0], parse_dates=["Date"]).set_index("Date")
    vcols = [c for c in vel.columns if "_vel" in c or "decomposed" in c]
    v = vel[vcols].mean(axis=1)
    rain = hm["R_1"]
    return v, rain


def month_wet(daily):
    mon = daily.groupby(daily.index.month)["rain"].mean()
    return mon >= mon.mean()


def event_table(v, rain):
    wet = (rain >= 1.0).values
    api7 = rain.rolling(7, min_periods=1).sum().shift(1).fillna(0.0)
    idx = rain.index
    rows, ev, in_ev = [], 0, False
    vv = v.reindex(idx)
    for i, w in enumerate(wet):
        if w and not in_ev:
            ev, in_ev = ev + 1, True
        elif not w:
            in_ev = False
    ev_id = np.cumsum(wet & np.concatenate(([True], ~wet[:-1]))) if len(wet) else np.zeros(0, int)
    for e in range(1, (ev_id.max() if len(ev_id) else 0) + 1):
        ii = np.where(ev_id == e)[0]
        s, t = ii[0], ii[-1]
        base = np.nanmean(vv.iloc[max(0, s - 7):s]) if s > 0 else vv.iloc[s]
        post = vv.iloc[t + 1:t + 4]
        if len(post) == 0 or not np.isfinite(base):
            continue
        delta = float(np.nanmax(post.values) - base)
        if not np.isfinite(delta):
            continue
        rows.append(dict(rsum=float(rain.iloc[ii].sum()), dur=int(t - s + 1),
                         iint=float(rain.iloc[ii].sum() / (t - s + 1)),
                         api7=float(api7.iloc[s]), delta=delta,
                         doy=int(idx[s].dayofyear)))
    return pd.DataFrame(rows)


def metrics(y, a):
    tp = int(((y == 1) & (a == 1)).sum()); fp = int(((y == 0) & (a == 1)).sum())
    tn = int(((y == 0) & (a == 0)).sum()); fn = int(((y == 1) & (a == 0)).sum())
    return dict(alerts=tp + fp, precision=round(tp / (tp + fp), 3) if tp + fp else None,
                recall=round(tp / (tp + fn), 3) if tp + fn else None,
                false_alarm_rate=round(fp / (fp + tn), 3) if fp + tn else None,
                miss_rate=round(fn / (tp + fn), 3) if tp + fn else None)


out = {}
for site, key in SITES.items():
    v, rain = load_site(key)
    v = v.dropna()
    rain = rain.dropna()
    df = pd.DataFrame({"v": v}).join(rain.rename("rain"), how="inner").dropna()
    # daily-layer ladder
    dv = df["v"].diff()
    corr_v = float(df["v"].corr(df["rain"])); corr_dv = float(dv.corr(df["rain"]))
    wet = (df["rain"] >= 1.0).astype(int).values
    dvb = (dv > dv.quantile(0.75)).astype(int).values[1:]
    te_r2d = temod.te_profile(wet[1:], dvb, lags=range(0, 11), n_surr=200, seed=10)
    te_best_lag = max(te_r2d, key=lambda L: (te_r2d[L]["z"] if np.isfinite(te_r2d[L]["z"]) else -9))
    te_peak = te_r2d[te_best_lag]
    # CCM: cross-map rain from the velocity-increment manifold (gap-free longest run)
    x = dv.dropna().values
    r_al = df["rain"].reindex(dv.dropna().index).values
    lagged = ccm.lagged_ccm(x, r_al, E=3, lags=range(0, 31))
    ccm_best = max(lagged, key=lambda L: lagged[L])
    # events x response variants
    ev = event_table(df["v"], df["rain"])
    print(f"  [debug] events built: {len(ev)}", flush=True)
    if len(ev) == 0:
        out[site] = {"error": "no events built"}
        continue
    ev["gate"] = ((ev["api7"] >= ev["api7"].median())
                  & np.isin(((ev["doy"] - 1) // 30 + 1).clip(1, 12).values,
                            np.where(month_wet(df.rename(columns={"v": "x"})).values)[0] + 1)).astype(int)
    q75 = float(ev["delta"].quantile(0.75))
    res = {"profile": dict(n_events=int(len(ev)), q75_delta=round(q75, 6),
                           corr_rain_vel=round(corr_v, 3), corr_rain_dvel=round(corr_dv, 3)),
           "ccm": {"best_lag": int(ccm_best), "skill": round(float(lagged[ccm_best]), 3)},
           "te_rain_to_burst": {"best_lag": int(te_best_lag), "z": round(float(te_peak["z"]), 1),
                                "p": round(float(te_peak["p"]), 4)}}
    for tag, ydef in (("A_delta_pos", (ev["delta"] > 0).astype(int).values),
                      ("B_burst_q75", (ev["delta"] > q75).astype(int).values)):
        X = np.c_[ev["iint"], ev["dur"], ev["iint"] * ev["dur"]]
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
        try:
            auc = round(float(np.mean(cross_val_score(pipe, X, ydef, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                                      scoring="roc_auc"))), 3)
        except Exception:
            auc = None
        pipe.fit(X, ydef)
        alert = (pipe.predict_proba(X)[:, 1] >= 0.5).astype(int)
        base = metrics(ydef, alert)
        gated = metrics(ydef, ((alert == 1) & (ev["gate"].values == 1)).astype(int))
        # attribution on full feature set
        Xf = np.c_[ev["iint"], ev["dur"], ev["api7"],
                   np.sin(2 * np.pi * ev["doy"] / 365), np.cos(2 * np.pi * ev["doy"] / 365)]
        pf = Pipeline([("sc", StandardScaler()),
                       ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))]).fit(Xf, ydef)
        coef = dict(zip(["I", "D", "API7", "sin_doy", "cos_doy"],
                        [round(float(c), 2) for c in pf.named_steps["lr"].coef_[0]]))
        res[tag] = {"cv_auc_ID": auc, "baseline": base, "gated": gated,
                    "std_logistic_coef": coef}
    out[site] = res
    print(f"=== {site} === n_ev={res['profile']['n_events']} corr(rain,vel)={res['profile']['corr_rain_vel']}")
    print(f"  CCM best lag={res['ccm']['best_lag']} skill={res['ccm']['skill']} | TE burst peak lag={res['te_rain_to_burst']['best_lag']} z={res['te_rain_to_burst']['z']} p={res['te_rain_to_burst']['p']}")
    for tag in ("A_delta_pos", "B_burst_q75"):
        r = res[tag]
        print(f"  {tag}: AUC={r['cv_auc_ID']} base={r['baseline']} | gated={r['gated']}")
    print(f"  coefs: {res['A_delta_pos']['std_logistic_coef']}")

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "french_pipeline_test.json")
json.dump(out, open(path, "w"), indent=1)
print("saved:", path)
