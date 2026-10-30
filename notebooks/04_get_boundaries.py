"""
step 4a: download toronto's postal area (fsa) shapes from statistics canada.

run: python notebooks/04_get_boundaries.py
out: data/processed/toronto_fsa.geojson, data/processed/fsa_centroids.csv
"""

import json
import math
import urllib.parse
import urllib.request

import pandas as pd

# statcan 2021 census fsa boundaries (cartographic version, coastline-clipped)
URL = "https://geo.statcan.gc.ca/geo_wa/rest/services/2021/Cartographic_boundary_files/MapServer/14/query"
params = {
    "where": "CFSAUID LIKE 'M%'",      # toronto areas all start with M
    "outFields": "CFSAUID",
    "outSR": "4326",                   # lat/lon
    "maxAllowableOffset": "0.0003",    # simplify shapes so the map loads fast
    "geometryPrecision": "5",
    "f": "geojson",
}
print("Downloading FSA boundaries from Statistics Canada...")
with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(params), timeout=60) as r:
    gj = json.load(r)

# rename the id field to fsa so it matches our tables
for f in gj["features"]:
    f["properties"] = {"fsa": f["properties"]["CFSAUID"]}
    f["id"] = f["properties"]["fsa"]
print(f"Got {len(gj['features'])} areas")

with open("data/processed/toronto_fsa.geojson", "w") as fh:
    json.dump(gj, fh)


def ring_centroid(ring):
    # area-weighted centroid of one polygon ring, in lon/lat
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    a /= 2
    if a == 0:
        return 0.0, ring[0][0], ring[0][1]
    return abs(a), cx / (6 * a), cy / (6 * a)


rows = []
for f in gj["features"]:
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    # use the biggest piece (islands are small)
    area, lon, lat = max(ring_centroid(p[0]) for p in polys)
    rows.append({"fsa": f["id"], "lon": lon, "lat": lat})

cent = pd.DataFrame(rows).sort_values("fsa")

# convert to km on a flat grid (fine at city scale)
LAT0 = 43.7
cent["x_km"] = cent["lon"] * 111.32 * math.cos(math.radians(LAT0))
cent["y_km"] = cent["lat"] * 110.57
cent.to_csv("data/processed/fsa_centroids.csv", index=False)
print("Saved: toronto_fsa.geojson, fsa_centroids.csv")

# check every area in our call data has a shape
calls = pd.read_pickle("data/processed/zone_blocks.pkl")["fsa"].unique()
missing = sorted(set(calls) - set(cent["fsa"]))
print(f"Areas in call data without a shape: {missing if missing else 'none'}")