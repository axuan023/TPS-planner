
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner import coverage, distance_matrix, nearest_km, optimize_ambulances, place_ambulances

# settings
ambulances = [10, 20]
forecasts = ["gbm", "hist_avg", "last_4_weeks", "nn"]

placers = {
    "optimizer": optimize_ambulances,
    "simple rule": place_ambulances
}

# load data
fc = pd.read_pickle("data/processed/forecast_2025.pkl")
forecasts = [f for f in forecasts if f in fc.columns]  # skip models not trained yet
cent = pd.read_csv("data/processed/fsa_centroids.csv")

cent = cent[cent["fsa"].isin(fc["fsa"].unique())]
cent = cent.sort_values("fsa").reset_index(drop=True)

areas = cent["fsa"].tolist()
dist = distance_matrix(cent)

# organize data by time and area
wide = {}

for col in forecasts + ["calls", "urgent_calls"]:
    wide[col] = fc.pivot(
        index="block", columns="fsa", values=col
    ).reindex(columns=areas)

# skip windows with no data
blocks = wide["calls"].index[
    wide["calls"].notna().any(axis=1)
]

print(f"Replaying {len(blocks)} windows...")

results = []
start = time.time()

# replay each 3-hour window
for i, block in enumerate(blocks):

    calls = wide["calls"].loc[block].fillna(0).to_numpy()
    urgent = wide["urgent_calls"].loc[block].fillna(0).to_numpy()

    for n in ambulances:
        for name, placer in placers.items():

            plans = {}

            # place ambulances using each forecast
            for forecast in forecasts:
                predicted = wide[forecast].loc[block].fillna(0).to_numpy()
                plans[forecast] = placer(predicted, dist, n)

            # compare with actual calls
            plans["hindsight"] = placer(calls, dist, n)

            for forecast, picks in plans.items():

                total_calls = calls.sum()
                total_urgent = urgent.sum()

                covered = coverage(calls, dist, picks) * total_calls

                urgent_covered = 0
                if total_urgent > 0:
                    urgent_covered = coverage(urgent, dist, picks) * total_urgent

                km_calls = nearest_km(calls, dist, picks) * total_calls

                results.append({
                    "block": block,
                    "ambulances": n,
                    "placement": name,
                    "forecast": forecast,
                    "calls": total_calls,
                    "covered": covered,
                    "urgent": total_urgent,
                    "urgent_covered": urgent_covered,
                    "km_x_calls": km_calls
                })

    # show progress
    if (i + 1) % 500 == 0:
        print(f"{i + 1} / {len(blocks)} windows done")


# save replay results
r = pd.DataFrame(results)
r.to_csv("data/processed/replay_2025.csv", index=False)

# calculate yearly totals
groups = ["ambulances", "placement", "forecast"]

totals = r.groupby(groups)[
    ["calls", "covered", "urgent", "urgent_covered", "km_x_calls"]
].sum()

summary = pd.DataFrame({
    "calls reached": totals["covered"] / totals["calls"],
    "urgent reached": totals["urgent_covered"] / totals["urgent"],
    "avg km to ambulance": totals["km_x_calls"] / totals["calls"]
}).reset_index()

summary.to_csv("data/processed/replay_summary.csv", index=False)


# print results
print(f"\nDone in {time.time() - start:.0f} seconds\n")

for n in ambulances:
    print(f"{n} AMBULANCES")

    table = summary[summary["ambulances"] == n].copy()

    table["calls reached"] = table["calls reached"].map("{:.1%}".format)
    table["urgent reached"] = table["urgent reached"].map("{:.1%}".format)
    table["avg km to ambulance"] = table["avg km to ambulance"].map("{:.2f}".format)

    print(table.drop(columns="ambulances").to_string(index=False))
    print()
