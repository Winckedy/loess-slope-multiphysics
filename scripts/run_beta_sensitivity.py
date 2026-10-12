"""Beta-sensitivity check for the WGCNA-style sensor network (paper1 v3).

Question: does the network layer's conclusion (tilt module coherent,
deformation-module-vs-rain coupling) depend on the soft-threshold power beta?
Recorded scale-free R2 was NaN/NaN/7.5e-34 at 5-6 nodes, so beta=6 was a
fallback. Here we recompute adjacency, TOM, module partition, and the
deformation-module vs rain trait correlation for beta in {1,2,4,6,8,12}.
"""
import os
import json, os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BETAS = [1, 2, 4, 6, 8, 12]


def tom_matrix(adj):
    n = len(adj)
    tom = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            k_products = adj[i] * adj[j]
            k_products[i] = k_products[j] = 0.0
            tom[i, j] = (np.sum(k_products) + adj[i, j]) / (
                min(adj[i].sum(), adj[j].sum()) + 1.0 - adj[i, j])
    return tom


def modules_from_tom(tom, sensors):
    """5-6 nodes: cut the 1-TOM hierarchical tree at depth 2 (sane for tiny n)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = 1.0 - tom
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0, None)
    link = linkage(squareform(dist, checks=False), method="average")
    return fcluster(link, t=2, criterion="maxclust")


def module_rain_corr(members, sensors, corr):
    """Eigengene (PC1) of the deformation (tilt) module vs rain."""
    tilt_idx = [i for i, s in enumerate(sensors) if s.startswith("tilt")]
    if len(tilt_idx) < 2:
        return np.nan
    sub = corr[np.ix_(tilt_idx, tilt_idx)]
    w, v = np.linalg.eigh(sub)
    eg = v[:, -1]
    row_mean = sub.mean(axis=1)
    if eg @ row_mean < 0:
        eg = -eg
    rain_idx = [i for i, s in enumerate(sensors) if s == "rain"][0]
    return float(eg @ corr[rain_idx, tilt_idx])


def modularity_proxy(adj, members):
    """Mean within-module adjacency minus mean between-module adjacency."""
    n = len(adj)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if members[i] == members[j] else between).append(adj[i, j])
    return (np.mean(within) if within else np.nan) - (np.mean(between) if between else np.nan)


print(f"{'site':<10}{'beta':>5}{'n_mod':>6}{'tilt-in-1-mod':>14}"
      f"{'tilt-vs-rain r':>15}{'modularity':>12}")
results = {}
for site in ("Xiji", "Qingyang", "Tianshui"):
    df = pd.read_csv(f"{BASE}/results/daily_{site}.csv")
    sensors = [c for c in ("rain", "pwp", "tiltx", "tilty", "tilt_mag",
                           "moist_mean", "water_level") if c in df.columns]
    d = df[sensors].dropna()
    corr = d.corr().values
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    rows = []
    for b in BETAS:
        adj = corr ** b
        tom = tom_matrix(adj)
        mem = modules_from_tom(tom, sensors)
        tilt_mods = {mem[i] for i, s in enumerate(sensors) if s.startswith("tilt")}
        r = module_rain_corr(mem, sensors, corr)
        mod = modularity_proxy(adj, mem)
        rows.append((b, len(set(mem)), len(tilt_mods) == 1, r, mod))
        print(f"{site:<10}{b:>5}{len(set(mem)):>6}{str(len(tilt_mods) == 1):>14}"
              f"{r:>15.3f}{mod:>12.3f}")
    results[site] = [{"beta": b, "n_modules": nm, "tilt_single_module": ts,
                      "module_rain_r": r, "modularity": m}
                     for b, nm, ts, r, m in rows]

out = f"{BASE}/results/loess_network_beta_sensitivity.json"
json.dump(results, open(out, "w"), indent=1)
print("\nsaved:", out)
