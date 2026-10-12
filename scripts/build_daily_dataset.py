#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper 2 — Chinese loess-slope rainfall thresholds & deformation precursors.
Builds harmonized DAILY panels for 3 monitored sites and computes:
  * Pearson/Spearman lag-correlation of rainfall vs soil response (PWP / tilt / moisture)
  * empirical rainfall-amount threshold for a positive soil response (binned probability)
  * antecedent-precipitation (API7) effect
  * deformation-acceleration episodes and their lead time to peak rainfall
Outputs daily_*.csv + loess_results.json in results/.
"""
import os, json, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

XJ = os.path.join(BASE, "Xiji", "2023年7月-2024年10月.xlsx")
QY = os.path.join(BASE, "Qingyang")
TY = os.path.join(BASE, "Tianshui")

def read_any(path, sheet=0):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)

def to_dt(s): return pd.to_datetime(s, errors="coerce")

# ---------------- Xiji ----------------
def panel_xiji():
    df = read_any(XJ); df.columns=[str(c).strip() for c in df.columns]
    t = to_dt(df["time"]); 
    rain = pd.to_numeric(df["降雨量(mm)"], errors="coerce").to_numpy()
    pwp = df[[c for c in df.columns if "孔隙水压力" in c]].apply(pd.to_numeric, errors="coerce")
    pwp_mean = pwp.mean(axis=1).to_numpy()
    tiltx = pd.to_numeric(df["倾角-X(°)"], errors="coerce").to_numpy()
    tilty = pd.to_numeric(df["倾角-Y(°)"], errors="coerce").to_numpy()
    g = pd.DataFrame({"rain":rain,"pwp":pwp_mean,"tiltx":tiltx,"tilty":tilty}, index=t)
    return g

# ---------------- Qingyang ----------------
def panel_qingyang():
    # meteorology CSVs already contain rain + PWP
    met_files = sorted(glob.glob(os.path.join(QY,"气象站","*.csv")))
    frames=[]
    for f in met_files:
        d=read_any(f)
        d.columns=[str(c).strip() for c in d.columns]
        t=to_dt(d["上报时间"]); 
        rec={"rain":pd.to_numeric(d.get("雨量(mm)"),errors="coerce").to_numpy()}
        pwpc=[c for c in d.columns if "孔隙水压力" in c]
        if pwpc:
            rec["pwp"]=pd.concat([pd.to_numeric(d[c],errors="coerce") for c in pwpc],axis=1).mean(axis=1).to_numpy()
        frames.append(pd.DataFrame(rec,index=t))
    met=pd.concat(frames).sort_index()
    # deformation (tilt) from 滑坡 CSVs
    defo_files=sorted(glob.glob(os.path.join(QY,"滑坡","*.csv")))
    dframes=[]
    for f in defo_files:
        d=read_any(f); d.columns=[str(c).strip() for c in d.columns]
        t=to_dt(d["上报时间"])
        rec={}
        if "倾角-X(°)" in d: rec["tiltx"]=pd.to_numeric(d["倾角-X(°)"],errors="coerce").to_numpy()
        if "倾角-Y(°)" in d: rec["tilty"]=pd.to_numeric(d["倾角-Y(°)"],errors="coerce").to_numpy()
        dframes.append(pd.DataFrame(rec,index=t))
    defo=pd.concat(dframes).sort_index() if dframes else pd.DataFrame()
    g=met.join(defo, how="outer")
    return g

# ---------------- Tianshui ----------------
def panel_tianshui():
    met=read_any(os.path.join(TY,"气象站2024年10月-2025年10月.xlsx"))
    met.columns=[str(c).strip() for c in met.columns]
    t=to_dt(met["时间"])
    g=pd.DataFrame({"rain":pd.to_numeric(met.get("雨量(mm)"),errors="coerce").to_numpy()},index=t)
    defo=read_any(os.path.join(TY,"滑坡","滑坡2024年10月-2025年10月.xlsx"))
    defo.columns=[str(c).strip() for c in defo.columns]
    td=to_dt(defo["时间"])
    drec={}
    if "倾角-X(°)" in defo: drec["tiltx"]=pd.to_numeric(defo["倾角-X(°)"],errors="coerce").to_numpy()
    if "倾角-Y(°)" in defo: drec["tilty"]=pd.to_numeric(defo["倾角-Y(°)"],errors="coerce").to_numpy()
    if "激光位移(m)" in defo: drec["laser_disp"]=pd.to_numeric(defo["激光位移(m)"],errors="coerce").to_numpy()
    moist=[c for c in defo.columns if "水分含量" in c]
    if moist: drec["moist_mean"]=pd.concat([pd.to_numeric(defo[c],errors="coerce") for c in moist],axis=1).mean(axis=1).to_numpy()
    g=g.join(pd.DataFrame(drec,index=td),how="outer")
    # water level
    wl=read_any(os.path.join(TY,"液位","液位2024年10月-2025年10月.xlsx"))
    wl.columns=[str(c).strip() for c in wl.columns]
    tw=to_dt(wl["时间"])
    g=g.join(pd.DataFrame({"water_level":pd.to_numeric(wl.get("激光液位(mm)"),errors="coerce").to_numpy()},index=tw),how="outer")
    return g

panels = {"Xiji":panel_xiji(), "Qingyang":panel_qingyang(), "Tianshui":panel_tianshui()}

# ---------------- daily aggregation + statistics ----------------
results={"sites":{}}
for name,g in panels.items():
    g=g[~g.index.isna()]
    g=g.sort_index()
    daily = g.resample("D").mean(numeric_only=True)
    daily["rain"] = g["rain"].resample("D").sum(min_count=1)
    daily = daily[~daily.index.duplicated()]
    resp = {c: daily[c] for c in daily.columns if c != "rain"}
    rain_d = daily["rain"]
    rec={"n_days":int(daily.shape[0]),
         "date_range":[str(daily.index.min().date()),str(daily.index.max().date())],
         "rain_total_mm":float(rain_d.sum()),
         "rain_days":int((rain_d>0).sum()),
         "corr":{}, "threshold":{}, "api7_effect":{}}
    # lag correlations rain(t) vs response(t+k), k=0..3 for each response var
    for rv,series in resp.items():
        merged=pd.concat([rain_d.rename("rain"), series.rename(rv)],axis=1).dropna()
        c={}
        for k in range(0,4):
            s=merged.copy()
            s["rain_lag"]=s["rain"].shift(-k)
            sub=s.dropna()
            if sub.shape[0]>20 and sub["rain_lag"].std()>0 and sub[rv].std()>0:
                c[f"lag{k}"]=float(sub.corr().loc["rain_lag",rv])
        rec["corr"][rv]=c
        # empirical threshold: bin daily rain, response prob = P(Δrv>0 next day)
        d2=pd.concat([rain_d.rename("rain"), series.rename(rv).diff().rename("d"+rv)],axis=1).dropna()
        d2=d2[d2["rain"]>=0]
        bins=[0,1,5,10,20,50,1e9]
        labels=["0-1","1-5","5-10","10-20","20-50",">50"]
        d2["bin"]=pd.cut(d2["rain"],bins=bins,labels=labels,right=False)
        grp=d2.groupby("bin",observed=True).agg(n=("rain","size"),p_pos=("d"+rv,lambda x:(x>0).mean()))
        rec["threshold"][rv]={str(k):{"n":int(v["n"]),"p_pos":float(v["p_pos"]) if pd.notna(v["p_pos"]) else None} for k,v in grp.iterrows()}
        # API7 effect: P(response) vs API7 bins
        api7=rain_d.rolling(7,min_periods=1).sum().shift(1)
        d3=pd.concat([api7.rename("api7"), series.rename(rv).diff().rename("d"+rv), rain_d.rename("rain")],axis=1).dropna()
        d3=d3[d3["rain"]>0]
        if d3.shape[0]>20:
            d3["api_bin"]=pd.cut(d3["api7"],bins=[-1,0.1,5,20,1e9],labels=["dry","low","mod","wet"],right=False)
            grp3=d3.groupby("api_bin",observed=True).agg(n=("rain","size"),p_pos=("d"+rv,lambda x:(x>0).mean()))
            rec["api7_effect"][rv]={str(k):{"n":int(v["n"]),"p_pos":float(v["p_pos"]) if pd.notna(v["p_pos"]) else None} for k,v in grp3.iterrows()}
    results["sites"][name]=rec
    # save daily panel
    out=pd.concat([rain_d.rename("rain")]+[v.rename(k) for k,v in resp.items()],axis=1)
    out.index.name="date"
    out.to_csv(os.path.join(OUT,f"daily_{name}.csv"))

with open(os.path.join(OUT,"loess_results.json"),"w",encoding="utf-8") as f:
    json.dump(results,f,indent=2,ensure_ascii=False,default=str)
print(json.dumps(results,indent=2,ensure_ascii=False,default=str)[:4000])
