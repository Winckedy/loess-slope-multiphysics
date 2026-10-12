# -*- coding: utf-8 -*-
"""Final raw site-map assets. All maps: square 1:1 output, true metric-square
geographic extent, NO red site markers (user annotates manually).

Per station (analysis/site_maps/<station>/):
  satellite_z18_1km.png   Esri World Imagery, 1 km x 1 km, light graticule
  hillshade_z15_1km.png   grey shaded relief, 1 km x 1 km
  osm_base_3km.png        OSM vector sketch (waterways/water/place), 3 km x 3 km
  regional_10km.png       regional sketch: DEM green tint + waterways, 10 km x 10 km

Attribution:
  satellite: Esri, Maxar, Earthstar Geographics, and the GIS User Community
  terrain:   AWS Open Data Terrain Tiles (formerly Mapzen)
  vectors:   (c) OpenStreetMap contributors (ODbL)
"""
import io
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

STATIONS = [
    ("Xiji",     106.7672, 36.1558),
    ("Qingyang", 107.665, 35.755),
    ("Tianshui", 105.710,  34.560),
]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "results", "site_maps")
CACHE = os.path.join(ROOT, "tile_cache")
SAT_TILE = ('https://server.arcgisonline.com/ArcGIS/rest/services/'
            'World_Imagery/MapServer/tile/%d/%d/%d')
DEM_TILE = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/%d/%d/%d.png'
OVERPASS = ('https://overpass-api.de/api/interpreter',
            'https://overpass.kumi.systems/api/interpreter')
OTM_TILE = 'https://a.tile.opentopomap.org/%d/%d/%d.png' 
FIGSIZE = (4.6, 4.6)   # square -> square output
AZ, ALT = 315.0, 45.0

plt.rcParams.update({"font.family": "Arial", "axes.linewidth": 0.7})


def http_get(url, timeout=60):
    import hashlib
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, hashlib.md5(url.encode()).hexdigest() + '.bin')
    if os.path.exists(cp):
        return open(cp, 'rb').read()
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'acad-map/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = r.read()
            open(cp, 'wb').write(d)
            return d
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}") from last


def download(urls):
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(http_get, urls))


def webmerc(lon, lat, z):
    x = (lon + 180.0) / 360.0 * (2 ** z)
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * (2 ** z)
    return x, y


def metric_box(lon, lat, z, half_m):
    """tile range + exact metric crop (true half_m x half_m metres).

    NOTE (Web Mercator property): locally, BOTH x and y directions convert
    metres->degrees as deg = m / (111320*cos(lat)); using 1/110574 for the
    latitude axis ignores the Mercator cos(lat) stretch and yields a map
    whose vertical extent is only cos(lat) of the requested metric size.
    """
    cx, cy = webmerc(lon, lat, z)
    lat_dpm = 1.0 / (1.11320e5 * math.cos(math.radians(lat)))
    lon_dpm = 1.0 / (1.11320e5 * math.cos(math.radians(lat)))
    tile_ppd = (360.0 / (2 ** z)) / 256.0          # deg per pixel
    hpx_lat = half_m * lat_dpm / tile_ppd
    hpx_lon = half_m * lon_dpm / tile_ppd
    xt, yt = int(math.floor(cx)), int(math.floor(cy))
    x0 = xt + int(math.floor(((cx - xt) * 256 - hpx_lon) / 256))
    x1 = xt + int(math.floor(((cx - xt) * 256 + hpx_lon) / 256))
    y0 = yt + int(math.floor(((cy - yt) * 256 - hpx_lat) / 256))
    y1 = yt + int(math.floor(((cy - yt) * 256 + hpx_lat) / 256))
    return (x0, x1, y0, y1,
            (cx - x0) * 256, (cy - y0) * 256, hpx_lon, hpx_lat, z)


def deg_extent(x0, x1, y0, y1, cxg, cyg, hpx_lon, hpx_lat, z):
    tile_deg = 360.0 / (2 ** z)
    lon0 = (x0 + (cxg - hpx_lon) / 256.0) * tile_deg - 180.0
    lon1 = (x0 + (cxg + hpx_lon) / 256.0) * tile_deg - 180.0
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y0 + (cyg - hpx_lat) / 256.0) / (2 ** z)))))
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y0 + (cyg + hpx_lat) / 256.0) / (2 ** z)))))
    return lon0, lon1, lat0, lat1


def overpass(bbox):
    s, w, n, e = bbox
    q = (f'[out:json][timeout:180];('
         f'way["waterway"]({s},{w},{n},{e});'
         f'way["natural"="water"]({s},{w},{n},{e});'
         f');out geom;')
    import requests
    for ep in OVERPASS:
        try:
            r = requests.post(ep, data={'data': q}, timeout=180,
                              headers={'User-Agent': 'acad-map/1.0'})
            if r.status_code == 200:
                return r.json().get('elements', [])
        except Exception:
            continue
    return []


def sat_raster(lon, lat, half_m=500.0, z=18):
    x0, x1, y0, y1, cxg, cyg, hpx_lon, hpx_lat, zc = metric_box(lon, lat, z, half_m)
    urls = [SAT_TILE % (z, ty, tx)
            for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    data = download(urls)
    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3), np.uint8)
    k = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            mosaic[(ty - y0) * 256:(ty - y0 + 1) * 256,
                   (tx - x0) * 256:(tx - x0 + 1) * 256] = \
                np.array(Image.open(io.BytesIO(data[k])).convert('RGB'))
            k += 1
    r0 = int(round(cyg - hpx_lat)); r1 = int(round(cyg + hpx_lat))
    c0 = int(round(cxg - hpx_lon)); c1 = int(round(cxg + hpx_lon))
    img = mosaic[r0:r1, c0:c1]
    if img.shape[1] > 1400:
        w2 = 1400
        img = np.array(Image.fromarray(img).resize(
            (w2, round(img.shape[0] * w2 / img.shape[1])), Image.LANCZOS))
    ext = deg_extent(x0, x1, y0, y1, cxg, cyg, hpx_lon, hpx_lat, z)
    return img, ext


def dem_raster(lon, lat, half_m, z):
    x0, x1, y0, y1, cxg, cyg, hpx_lon, hpx_lat, zc = metric_box(lon, lat, z, half_m)
    urls = [DEM_TILE % (z, tx, ty)
            for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    data = download(urls)
    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.float64)
    k = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            a = np.array(Image.open(io.BytesIO(data[k])).convert('RGB'), np.float64)
            mosaic[(ty - y0) * 256:(ty - y0 + 1) * 256,
                   (tx - x0) * 256:(tx - x0 + 1) * 256] = \
                (a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0) - 32768.0
            k += 1
    r0 = int(round(cyg - hpx_lat)); r1 = int(round(cyg + hpx_lat))
    c0 = int(round(cxg - hpx_lon)); c1 = int(round(cxg + hpx_lon))
    d = np.where(mosaic[r0:r1, c0:c1] < 0, 0.0, mosaic[r0:r1, c0:c1])
    ext = deg_extent(x0, x1, y0, y1, cxg, cyg, hpx_lon, hpx_lat, z)
    return d, ext


def hillshade_arr(d):
    gy, gx = np.gradient(d)
    gy *= 1.6; gx *= 1.6
    sl = np.sqrt(gx ** 2 + gy ** 2)
    asp = np.arctan2(-gx, gy)
    az, alt = AZ, ALT
    sh = (np.sin(alt) * np.cos(np.arctan(sl)) +
          np.cos(alt) * np.sin(np.arctan(sl)) * np.cos(az - asp - math.pi / 2))
    s = np.clip((sh + 1) / 2, 0, 1)
    lo, hi = float(s.min()), float(s.max())
    return np.clip((s - lo) / (hi - lo) * 0.65 + 0.35, 0.35, 1.0)


def safe_save(fig, out, dpi=300):
    try:
        fig.savefig(out, dpi=dpi)
        print('   saved', os.path.basename(out))
    except PermissionError:
        print('   SKIP (file locked, close the viewer):', out)


def finish(fig, ax, out, lon_range, lat_range, coslat, grid=True):
    ax.set_xlim(*lon_range); ax.set_ylim(*lat_range)
    ax.set_aspect(1.0 / coslat)                    # true-metric, undistorted
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlabel('Longitude (°E)', fontsize=8)
    ax.set_ylabel('Latitude (°N)', fontsize=8)
    if grid:
        ax.grid(True, color='#dddddd', lw=0.4, zorder=3)
    try:
        fig.savefig(out, dpi=300)
        print("   saved", os.path.basename(out))
    except PermissionError:
        print("   SKIP (file locked by a viewer, close it and rerun):", out)
    plt.close(fig)
    print("  ", os.path.basename(out))


# ----------------------------------------------------------------- stations
for name, lon, lat in STATIONS:
    sdir = os.path.join(OUTDIR, name)
    os.makedirs(sdir, exist_ok=True)
    coslat = math.cos(math.radians(lat))
    print(name)

    # 1. satellite 1 km ---------------------------------------------------
    img, (lo0, lo1, la0, la1) = sat_raster(lon, lat, half_m=1500.0, z=16)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(img, extent=(lo0, lo1, la0, la1), zorder=1)
    finish(fig, ax, os.path.join(sdir, "satellite_z16_3km.png"),
           (lo0, lo1), (la0, la1), coslat, grid=True)

    # 2. hillshade 1 km ---------------------------------------------------
    d1, (lo0, lo1, la0, la1) = dem_raster(lon, lat, half_m=1500.0, z=14)
    d1 = gaussian_filter(d1, 1.0)
    d1 = np.array(Image.fromarray(d1.astype(np.float32)).resize(
        (d1.shape[1] * 6, d1.shape[0] * 6), Image.BICUBIC), np.float64)
    d1 = gaussian_filter(d1, 1.0)
    hs = hillshade_arr(d1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(hs, cmap='gray', vmin=0, vmax=1,
              extent=(lo0, lo1, la0, la1), zorder=1)
    finish(fig, ax, os.path.join(sdir, "hillshade_z14_3km.png"),
           (lo0, lo1), (la0, la1), coslat, grid=False)

    # 3. OSM vector sketch 3 km -------------------------------------------
    half = 1500.0
    lon_half = half * (1.0 / (1.11320e5 * coslat))
    lat_half = half * (1.0 / 1.10574e5)
    bbox = (lat - lat_half, lon - lon_half, lat + lat_half, lon + lon_half)
    import requests
    q = (f'[out:json][timeout:120];('
         f'way["waterway"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});'
         f'way["natural"="water"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});'
         f'node["place"~"city|town"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});'
         f');out geom;')
    els = []
    for ep in OVERPASS:
        try:
            r = requests.post(ep, data={'data': q}, timeout=180,
                              headers={'User-Agent': 'acad-map/1.0'})
            if r.status_code == 200:
                els = r.json().get('elements', [])
                break
        except Exception:
            continue
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor('#edf0ea')
    for el in els:
        if el.get('type') != 'way' or 'geometry' not in el:
            continue
        t = el.get('tags', {})
        pts = [(g['lon'], g['lat']) for g in el['geometry']]
        if not pts:
            continue
        if 'natural' in t and t['natural'] == 'water':
            ax.add_patch(plt.Polygon(pts, closed=True, fc='#7fc7ea',
                                     ec='none', zorder=2))
        else:
            w = 1.1 if t.get('waterway') == 'river' else 0.6
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color='#7fb8dd', lw=w, zorder=2, solid_capstyle='round')
    finish(fig, ax, os.path.join(sdir, "osm_base_3km.png"),
           (bbox[1], bbox[3]), (bbox[0], bbox[2]), coslat, grid=False)

    # 3b. OpenTopoMap sketch 3 km -----------------------------------------
    ot_z = 15
    x0t, x1t, y0t, y1t, cxgt, cygt, hpxl, hpxa, _ = metric_box(lon, lat, ot_z, 1500.0)
    ot_urls = [OTM_TILE % (ot_z, tx, ty)
               for ty in range(y0t, y1t + 1) for tx in range(x0t, x1t + 1)]
    ot_mosaic = np.zeros(((y1t - y0t + 1) * 256, (x1t - x0t + 1) * 256, 3), np.uint8)
    k = 0
    for ty in range(y0t, y1t + 1):
        for tx in range(x0t, x1t + 1):
            ot_mosaic[(ty - y0t) * 256:(ty - y0t + 1) * 256,
                      (tx - x0t) * 256:(tx - x0t + 1) * 256] =                 np.array(Image.open(io.BytesIO(download(ot_urls)[k] if False else http_get(ot_urls[k]))).convert('RGB'))
            k += 1
    otr0 = int(round(cygt - hpxa)); otr1 = int(round(cygt + hpxa))
    otc0 = int(round(cxgt - hpxl)); otc1 = int(round(cxgt + hpxl))
    ot_img = ot_mosaic[otr0:otr1, otc0:otc1]
    ot_ext = deg_extent(x0t, x1t, y0t, y1t, cxgt, cygt, hpxl, hpxa, ot_z)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(ot_img, extent=(ot_ext[0], ot_ext[1], ot_ext[2], ot_ext[3]), zorder=1)
    finish(fig, ax, os.path.join(sdir, 'topo_base_3km.png'),
           (ot_ext[0], ot_ext[1]), (ot_ext[2], ot_ext[3]), coslat, grid=False)

print("ALL DONE ->", OUTDIR)
