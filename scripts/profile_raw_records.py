# -*- coding: utf-8 -*-
"""Step 1 dataset deep-dive: quantitative profiling of the three-station loess dataset."""
import glob, json, os, sys
import pandas as pd
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw_record_profile.json")

def read_csv_cn(path):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None

def profile_csv(paths, name, tcol="上报时间"):
    dfs = []
    for p in paths:
        df = read_csv_cn(p)
        if df is None:
            print(f"  !! unreadable {p}")
            continue
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).sort_values(tcol)
    df = df.drop_duplicates(subset=tcol)
    info = {
        "name": name, "n_rows": int(len(df)),
        "t_min": str(df[tcol].min()), "t_max": str(df[tcol].max()),
        "channels": list(df.columns),
    }
    # median sampling interval (hours)
    dt = df[tcol].diff().dt.total_seconds().dropna() / 60.0
    info["median_dt_min"] = float(dt.median())
    # gaps > 24h
    gaps = dt[dt > 1440]
    info["n_gaps_gt24h"] = int((gaps > 1440).sum())
    info["gap_days_total"] = float(gaps.sum() / 1440.0)
    # missingness of numeric channels
    miss = {}
    for c in df.columns:
        if c == tcol:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() == 0:
            miss[c] = {"n_nan": len(df), "std": None, "zeros": len(df)}
            continue
        miss[c] = {
            "nan_frac": round(float(v.isna().mean()), 4),
            "std": round(float(v.std()), 4),
            "zeros_frac": round(float((v == 0).mean()), 4),
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
        }
    info["channels_stat"] = miss
    return info

res = {}

# --- Qingyang Huoxianggou
QY = os.path.join(ROOT, "Qingyang")
res["QY_met"] = profile_csv(sorted(glob.glob(os.path.join(QY, "气象站", "*.csv"))), "QY_met_10min")
res["QY_soil"] = profile_csv(sorted(glob.glob(os.path.join(QY, "水体", "*.csv"))), "QY_soil_10min")
res["QY_slide"] = profile_csv(sorted(glob.glob(os.path.join(QY, "滑坡", "*.csv"))), "QY_slide_5min")

# --- Tianshui Luoyugou
TS = os.path.join(ROOT, "Tianshui")
res["TS_met"] = profile_csv(sorted(glob.glob(os.path.join(TS, "21609*.csv"))), "TS_met")
res["TS_level"] = profile_csv(sorted(glob.glob(os.path.join(TS, "液位", "*.csv"))), "TS_level_60min")
res["TS_slide"] = profile_csv(sorted(glob.glob(os.path.join(TS, "滑坡", "*.csv"))), "TS_slide_5min")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("saved", OUT)

# ---- compact summary
for k, v in res.items():
    if v is None:
        print(k, "NONE")
        continue
    print(f"\n### {k}: {v['n_rows']} rows, {v['t_min']} .. {v['t_max']}, dt={v['median_dt_min']}min, gaps>24h={v['n_gaps_gt24h']} ({v['gap_days_total']:.0f}d)")
    sm = v["channels_stat"]
    for c, s in sm.items():
        if "nan_frac" in s:
            flag = ""
            if s["std"] is not None and s["std"] < 1e-6:
                flag = " <CONSTANT>"
            print(f"  {c}: nan={s['nan_frac']}, zeros={s['zeros_frac']}, std={s['std']}, rng=[{s['min']},{s['max']}]{flag}")
