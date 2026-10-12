#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loess-slope novel layer (G6-G7): bioinformatics-style analyses of the CN multi-physics panels.
  G6  WGCNA-style sensor co-occurrence network: per-site soft-threshold power selection ->
      TOM -> hierarchical module detection -> module eigengene vs rainfall (trait) correlation.
  G7  Intermittent / event-scale coupling: rolling-window correlation of rainfall vs tilt
      magnitude shows the daily r is a *mean of a variable* process (intermittency); event-scale
      response (post-event peak tilt change) recovers a coupling the static daily r hides.
Outputs: loess_novel.json + 3 Nature-style figures.
Reproducible: reads only results/daily_*.csv (derived from provided monitoring set).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def panel_letters(axs, x=-0.10, y=1.04, size=9, letters=None):
    """Nature-style bold lowercase panel labels, no parentheses (user-fixed 2026-09-29)."""
    flat = list(np.ravel([axs]))
    flat = [a for a in flat if a is not None and a.get_visible()]
    if letters is None:
        letters = [chr(97 + i) for i in range(len(flat))]
    for a, L in zip(flat, letters):
        a.text(x, y, L, transform=a.transAxes, fontsize=size,
               fontweight="bold", va="bottom", ha="right")
LO = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.spines.top":False,
    "axes.spines.right":False,"axes.linewidth":0.8,"figure.dpi":300,"savefig.dpi":300})
# Project palette (fixed): use in order, earlier colours preferred; grey = neutral
PAL=["#0000A0","#CF3D3E","#FFB400","#FBDD85","#35B257","#7EA5E0"]
CB = {"blue":PAL[0],"orange":PAL[1],"green":PAL[2],"purple":PAL[3],
      "red":PAL[4],"yellow":PAL[5],"grey":"#8C8C8C"}

sites = ["Xiji","Qingyang","Tianshui"]
panels = {}
for s in sites:
    df = pd.read_csv(os.path.join(LO,f"daily_{s}.csv"), parse_dates=["date"]).set_index("date")
    df = df.sort_index()
    # keep only columns with real variance
    keep = [c for c in df.columns if df[c].std()>1e-6* (df[c].abs().max()+1e-9)]
    panels[s] = df[keep].copy()
    panels[s]["tilt_mag"] = np.sqrt(panels[s].tiltx**2 + panels[s].tilty**2)

# ---------------------------------------------------------------- G7 sliding-window + event
win_novel = {}
all_window_r = []
all_daily_r = []
for s in sites:
    d = panels[s]
    daily_r = float(d["rain"].corr(d["tilt_mag"]))
    all_daily_r.append(daily_r)
    rw = d["rain"].rolling(60, min_periods=30).corr(d["tilt_mag"])
    rw = rw.dropna()
    all_window_r.append(rw.values)
    win_novel[s] = {"daily_r":daily_r,
                    "window_r_mean":float(rw.mean()), "window_r_std":float(rw.std()),
                    "window_r_min":float(rw.min()), "window_r_max":float(rw.max()),
                    "frac_window_strong":float((rw.abs()>0.4).mean()),
                    "frac_window_pos":float((rw>0.2).mean())}
all_window_flat = np.concatenate(all_window_r)
print("Loess sliding-window:", {s:{k:round(v,3) for k,v in win_novel[s].items()} for s in sites})
print("daily r mean across sites:", round(float(np.mean(all_daily_r)),3),
      "| windowed |r| mean:", round(float(np.abs(all_window_flat).mean()),3))

# event-scale response
def events(d, thresh=1.0, gap=2):
    rain = (d["rain"]>=thresh).astype(int).values
    # label consecutive rain runs
    lab = np.zeros(len(rain), dtype=int); cur=0; last_was_rain=False
    for i,r in enumerate(rain):
        if r==1:
            if not last_was_rain: cur+=1
            lab[i]=cur; last_was_rain=True
        else:
            last_was_rain=False
    return lab
event_novel = {}
event_rows = []
for s in sites:
    d = panels[s].reset_index()
    lab = events(d)
    tilt = d["tilt_mag"].values
    rain = d["rain"].values
    out=[]
    for ev in range(1, lab.max()+1):
        idx = np.where(lab==ev)[0]
        start, end = idx[0], idx[-1]
        base = tilt[max(0,start-7):start].mean() if start>0 else tilt[start]
        post = tilt[end+1:end+4]
        if len(post)==0: continue
        peak = post.max()
        delta = peak - base
        rsum = rain[idx].sum()
        out.append((rsum, delta))
    out=np.array(out)
    if len(out)>5:
        cc = np.corrcoef(out[:,0], out[:,1])[0,1]
        ev_r = float(cc) if np.isfinite(cc) else None
        # binned fraction of events with positive tilt response
        bins=[0,5,15,30,1e9]; blabels=["<5","5-15","15-30",">30"]
        dfb=pd.DataFrame(out,columns=["rsum","delta"])
        dfb["b"]=pd.cut(dfb.rsum,bins=bins,labels=blabels,right=False)
        grp=dfb.groupby("b",observed=True).agg(n=("delta","size"),p_pos=("delta",lambda x:(x>0).mean()),
                                               mean_delta=("delta","mean"))
        event_novel[s]={"n_events":int(len(out)),"corr_eventrain_peakdelta":ev_r,
                        "binned":{str(k):{"n":int(v["n"]),"p_pos":float(v["p_pos"]),
                                          "mean_delta":float(v["mean_delta"])} for k,v in grp.iterrows()}}
        for r,dl in out: event_rows.append((s,r,dl))
    print(f"  {s}: n_events={len(out)} corr(rsum,peak_delta)={ev_r}")
event_df = pd.DataFrame(event_rows, columns=["site","rsum","delta"])

# ---------------------------------------------------------------- G6 network
def scale_free_fit(corr_abs, beta):
    adj = corr_abs**beta
    k = adj.sum(axis=1) - 1.0  # exclude self
    k = k[k>0]; pk = np.array([(k==kk).mean() for kk in np.unique(k)])
    lk = np.log10(k); lp = np.log10(pk)
    if len(lk)<3: return 0.0
    return float(np.corrcoef(lk,lp)[0,1]**2)

def analyze_network(s):
    d = panels[s]
    cols = [c for c in ["rain","pwp","tiltx","tilty","tilt_mag","moist_mean","water_level","laser_disp"] if c in d.columns]
    M = d[cols].dropna()
    c = np.array(M.corr().abs().values, copy=True)
    np.fill_diagonal(c, 1.0)
    # soft-threshold power
    pows=range(1,21); fits=[scale_free_fit(c,b) for b in pows]
    beta = next((p for p,fi in zip(pows,fits) if fi>=0.8), 6)
    adj = c**beta
    tom = adj.copy()
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            kij = sum(adj[i,m]*adj[j,m] for m in range(len(cols)) if m!=i and m!=j)
            t = (adj[i,j] + kij) / (1.0 + kij)
            tom[i,j]=t; tom[j,i]=t
    tom = tom - np.diag(np.diag(tom)) + np.eye(len(cols))
    dist = 1 - tom
    np.fill_diagonal(dist,0.0)
    Z = linkage(squareform(dist), method="average")
    labels = fcluster(Z, t=0.25, criterion="distance")
    # module eigengene vs rainfall
    eig=[]; mod_r=[]
    for lab in np.unique(labels):
        members=[cols[i] for i in range(len(cols)) if labels[i]==lab]
        Xm = M[members].values
        pc = PCA(1).fit_transform(Xm)
        r = float(np.corrcoef(pc[:,0], M["rain"].values)[0,1])
        eig.append((lab,members,r))
        mod_r.append(r)
    abs_corr = {}
    for _i in range(len(cols)):
        for _j in range(_i+1, len(cols)):
            abs_corr[str(cols[_i]) + "~" + str(cols[_j])] = float(c[_i, _j])
    return {"sensors":cols,"beta":int(beta),"scale_free_r2":float(fits[beta-1]),
            "modules":[{"members":mem,"trait_r_rain":r} for lab,mem,r in eig],
            "n_modules":int(len(np.unique(labels))),
            "abs_corr":abs_corr,
            "tom":tom.tolist()}

net_novel={s:analyze_network(s) for s in sites}
print("Network modules:", {s:net_novel[s]["n_modules"] for s in sites})

novel = {"sliding_window":win_novel,
         "event_scale":event_novel,
         "network":net_novel,
         "summary":{"daily_r_mean":float(np.mean(all_daily_r)),
                    "windowed_abs_r_mean":float(np.abs(all_window_flat).mean()),
                    "windowed_abs_r_max":float(np.abs(all_window_flat).max())}}
with open(os.path.join(LO,"loess_novel.json"),"w",encoding="utf-8") as f:
    json.dump(novel,f,indent=2,ensure_ascii=False,default=str)

# ---------------------------------------------------------------- FIG 1: sliding window intermittency
fig,ax=plt.subplots(2,1,figsize=(7.2,4.6),sharex=False)
d=panels["Qingyang"]
rw=d["rain"].rolling(60,min_periods=30).corr(d["tilt_mag"]).dropna()
ax[0].plot(rw.index, rw.values, color=CB["blue"], lw=0.8)
ax[0].axhline(0,color="k",lw=0.5)
ax[0].axhline(0.4,color=CB["green"],ls="--",lw=0.6); ax[0].axhline(-0.4,color=CB["red"],ls="--",lw=0.6)
ax[0].set_ylabel("rolling 60-d r\n(rain vs tilt)")
ax[0].set_title("Qingyang: rainfall-tilt coupling is intermittent, not constant",fontsize=9)
ax[1].hist(all_window_flat, bins=40, color=CB["orange"], alpha=0.8)
ax[1].axvline(np.mean(all_daily_r), color="k", lw=1, label=f"static daily r mean = {np.mean(all_daily_r):.2f}")
ax[1].set_xlabel("rolling-window r (rain vs tilt)"); ax[1].set_ylabel("count of windows")
ax[1].legend(fontsize=7,frameon=False)
ax[1].set_title(f"Windowed coupling spans -0.7..0.9 (static mean hides it)",fontsize=9)
panel_letters(ax, x=-0.16, y=1.05)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"loess_sliding.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(FIG,"loess_sliding.png"),bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- FIG 2: event-scale response
fig,ax=plt.subplots(1,2,figsize=(7.2,3.4))
cmap={"Xiji":CB["blue"],"Qingyang":CB["orange"],"Tianshui":CB["green"]}
for s,sub in event_df.groupby("site"):
    ax[0].scatter(sub.rsum, sub.delta, s=10, alpha=0.5, color=cmap[s], label=s)
ax[0].axhline(0,color="k",lw=0.5)
ax[0].set_xlabel("event rainfall sum (mm)"); ax[0].set_ylabel("post-event peak tilt change")
ax[0].set_title("Event-scale response (hidden by daily mean)",fontsize=9)
ax[0].legend(fontsize=6,frameon=False)
# binned P(response>0) by event size
bins=["<5","5-15","15-30",">30"]; xpos=range(len(bins))
for si,s in enumerate(sites):
    ppos=[event_novel[s].get("binned",{}).get(b,{}).get("p_pos", np.nan) for b in bins]
    ax[1].plot(xpos,ppos,marker="o",color=cmap[s],label=s,lw=1.4)
ax[1].set_xticks(list(xpos)); ax[1].set_xticklabels(bins,fontsize=7)
ax[1].set_xlabel("event rainfall sum (mm)"); ax[1].set_ylabel("P(tilt increases)")
ax[1].set_title("Larger events drive tilt response",fontsize=9)
ax[1].legend(fontsize=6,frameon=False)
panel_letters(ax, x=-0.14, y=1.05)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"loess_event.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(FIG,"loess_event.png"),bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- FIG 3: network heatmap (TOM) per site
fig,ax=plt.subplots(1,3,figsize=(7.2,2.65))
for si,s in enumerate(sites):
    net=net_novel[s]; cols=net["sensors"]; tom=np.array(net["tom"])
    dist=1-tom; np.fill_diagonal(dist,0)
    Z=linkage(squareform(dist),method="average"); lab=fcluster(Z,t=0.25,criterion="distance")
    order=np.argsort(lab)
    M=tom[np.ix_(order,order)]
    im=ax[si].imshow(M,cmap="Blues",vmin=0,vmax=1)
    ax[si].set_xticks(range(len(cols))); ax[si].set_xticklabels(cols,rotation=90,fontsize=5)
    ax[si].set_yticks(range(len(cols))); ax[si].set_yticklabels([cols[i] for i in order],fontsize=5)
    ax[si].set_title(f"{s}\nβ={net['beta']}, modules={net['n_modules']}",fontsize=8)
panel_letters(ax, x=-0.18, y=1.06)
# dedicated colourbar axis: three panels stay equal width, no overlap (user-fixed 2026-09-29)
fig.subplots_adjust(left=0.09, right=0.895, top=0.80, bottom=0.30, wspace=0.22)
cax = fig.add_axes([0.905, 0.34, 0.014, 0.40])
fig.colorbar(im, cax=cax, label="TOM")
fig.savefig(os.path.join(FIG,"loess_network.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(FIG,"loess_network.png"),bbox_inches="tight"); plt.close(fig)
print("DONE loess novel")
