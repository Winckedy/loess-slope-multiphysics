#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper 2 figures — Loess rainfall thresholds & precursors (Nature style)."""
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); L=r"results"; OUT=os.path.join(ROOT,"analysis","figures")
os.makedirs(OUT,exist_ok=True)
R=json.load(open(os.path.join(ROOT,L,"loess_results.json"),encoding="utf-8"))

plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.linewidth':0.8,
 'axes.spines.top':False,'axes.spines.right':False,'xtick.major.width':0.8,
 'ytick.major.width':0.8,'xtick.major.size':3,'ytick.major.size':3,
 'xtick.direction':'out','ytick.direction':'out','figure.dpi':300,
 'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.05,
 'pdf.fonttype':42,'ps.fonttype':42})
# Project palette (fixed): use in order, earlier colours preferred; grey/black = neutrals
PAL=["#0000A0","#CF3D3E","#FFB400","#FBDD85","#35B257","#7EA5E0"]
C={"blue":PAL[0],"red":PAL[1],"green":PAL[4],"orange":PAL[2],
   "purple":PAL[5],"teal":PAL[3],"cb_blue":PAL[0],"cb_orange":PAL[2],"gray":"#8C8C8C"}

fig=plt.figure(figsize=(7.2,5.4))
gs=gridspec.GridSpec(2,2,figure=fig,wspace=0.4,hspace=0.45)

# ---- Panel A: Xiji daily rainfall (bars) vs PWP (line) excerpt ----
axA=fig.add_subplot(gs[0,0])
xj=pd.read_csv(os.path.join(ROOT,L,"daily_Xiji.csv"),parse_dates=["date"]).dropna(subset=["rain","pwp"])
w=xj[(xj.date>="2024-04-01")&(xj.date<="2024-09-30")]
axA.bar(w.date, w.rain, color=C["cb_blue"], width=1.0, alpha=0.7, label="Rain (mm d/day)")
axA.set_ylabel("Rain (mm d/day)", color=C["cb_blue"]); axA.tick_params(axis='y',labelcolor=C["cb_blue"])
ax2=axA.twinx(); ax2.plot(w.date, w.pwp, color=C["red"], lw=1.3, label="PWP (kPa)")
ax2.set_ylabel("Pore-water pressure (kPa)", color=C["red"]); ax2.tick_params(axis='y',labelcolor=C["red"])
axA.set_title("a | Xiji: rainfall vs pore pressure (2024)", fontsize=8, loc="left", weight="bold")
axA.set_xticklabels([])

# ---- Panel B: rainfall-amount threshold for PWP response (Xiji) ----
axB=fig.add_subplot(gs[0,1])
thr=R["sites"]["Xiji"]["threshold"]["pwp"]
bins=list(thr.keys()); ps=[thr[b]["p_pos"] for b in bins]; ns=[thr[b]["n"] for b in bins]
axB.bar(range(len(bins)), ps, color=C["cb_orange"])
axB.set_xticks(range(len(bins))); axB.set_xticklabels(bins, rotation=30, ha="right", fontsize=7)
axB.set_ylabel("P(ΔPWP > 0 next day)"); axB.set_xlabel("Daily rainfall bin (mm)")
axB.set_title("b | Xiji PWP-response probability vs rainfall", fontsize=8, loc="left", weight="bold")
for i,(p,n) in enumerate(zip(ps,ns)): axB.annotate(f"n={n}",(i,p),textcoords="offset points",xytext=(0,5),ha="center",fontsize=6)

# ---- Panel C: cross-site daily correlation rain vs soil response ----
axC=fig.add_subplot(gs[1,0])
rows=[]; labels=[]
for site,rec in R["sites"].items():
    for rv,c in rec["corr"].items():
        l0=c.get("lag0")
        if l0 is not None: rows.append((f"{site[:3]}\n{rv}", l0))
names=[r[0] for r in rows]; vals=[r[1] for r in rows]
colors=[C["cb_blue"] if v>=0 else C["red"] for v in vals]
axC.barh(range(len(names)), vals, color=colors)
axC.set_yticks(range(len(names))); axC.set_yticklabels(names, fontsize=6)
axC.axvline(0,color="k",lw=0.6); axC.set_xlabel("Pearson r (daily, lag 0)")
axC.set_title("c | Rain–soil-response correlation by site", fontsize=8, loc="left", weight="bold")

# ---- Panel D: rainfall regime by site ----
axD=fig.add_subplot(gs[1,1])
sites=list(R["sites"].keys())
tot=[R["sites"][s]["rain_total_mm"] for s in sites]
days=[R["sites"][s]["n_days"] for s in sites]
ann=[t/d*365 for t,d in zip(tot,days)]
axD.bar(range(len(sites)), ann, color=C["teal"])
axD.set_xticks(range(len(sites))); axD.set_xticklabels([s[:4] for s in sites], fontsize=7)
axD.set_ylabel("Mean annual rainfall (mm yr/day)")
axD.set_title("d | Site rainfall regime", fontsize=8, loc="left", weight="bold")

for ax in [axA,axB,axC,axD]:
    ax.text(-0.16,1.04,"",transform=ax.transAxes,fontsize=8,weight="bold",va="top",ha="right")
fig.suptitle("Figure 1. Rainfall–soil-response coupling across three monitored loess slopes", fontsize=9, weight="bold", x=0.5, y=0.98)
plt.savefig(os.path.join(OUT,"paper2_fig1.pdf")); plt.savefig(os.path.join(OUT,"paper2_fig1.png"),dpi=300); plt.savefig(os.path.join(OUT,"paper2_fig1.svg"))
print("Saved paper2_fig1")
