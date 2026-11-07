
import json
import math
import urllib.parse
import urllib.request
import pandas as pd

# download boundaries
url = "https://geo.statcan.gc.ca/geo_wa/rest/services/2021/Cartographic_boundary_files/MapServer/14/query"

params = {
    "where": "CFSAUID LIKE 'M%'",
    "outFields": "CFSAUID",
    "outSR": "4326",
    "maxAllowableOffset": "0.0003",
    "geometryPrecision": "5",
    "f": "geojson"
}

print("Downloading FSA boundaries...")

link = url + "?" + urllib.parse.urlencode(params)

with urllib.request.urlopen(link, timeout=60) as response:
    data = json.load(response)

# rename area codes
for feature in data["features"]:
    fsa = feature["properties"]["CFSAUID"]
    feature["properties"] = {"fsa": fsa}
    feature["id"] = fsa

print(f"Found {len(data['features'])} areas")

# save boundaries
with open("data/processed/toronto_fsa.geojson", "w") as file:
    json.dump(data, file)


# find the centre of each area
def ring_centroid(ring):
    area = 0
    cx = 0
    cy = 0

    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]

        cross = x1 * y2 - x2 * y1

        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    area = area / 2

    if area == 0:
        return 0, ring[0][0], ring[0][1]

    lon = cx / (6 * area)
    lat = cy / (6 * area)

    return abs(area), lon, lat


# calculate area centres
rows = []

for feature in data["features"]:
    geometry = feature["geometry"]

    if geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        polygons = [geometry["coordinates"]]

    centres = []

    for polygon in polygons:
        centres.append(ring_centroid(polygon[0]))

    area, lon, lat = max(centres)

    rows.append({
        "fsa": feature["id"],
        "lon": lon,
        "lat": lat
    })

# save coordinates
cent = pd.DataFrame(rows)
cent = cent.sort_values("fsa")

# convert coordinates to km
lat0 = 43.7

cent["x_km"] = cent["lon"] * 111.32 * math.cos(math.radians(lat0))
cent["y_km"] = cent["lat"] * 110.57

cent.to_csv("data/processed/fsa_centroids.csv", index=False)

print("Saved boundaries and centroids")

# check for missing areas
calls = pd.read_pickle("data/processed/zone_blocks.pkl")
call_areas = calls["fsa"].unique()

missing = sorted(set(call_areas) - set(cent["fsa"]))

if missing:
    print("Missing areas:", missing)
else:
    print("All areas have boundaries")
