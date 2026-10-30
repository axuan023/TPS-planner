# replay all of 2025.

# run: .venv/Scripts/python notebooks/07_replay.py
# in:  data/processed/forecast_2025.pkl, data/processed/fsa_centroids.csv
# out: data/processed/replay_2025.csv (every window), data/processed/replay_summary.csv

# for every 3-hour window in 2025: place ambulances using each forecast,
# then score the plan on the calls that really happened.
# this gives the headline numbers for the write-up.

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so planner.py is found
from planner import coverage, distance_matrix, nearest_km, optimize_ambulances, place_ambulances

AMBULANCES = [10, 20]
FORECASTS = ["gbm", "hist_avg", "last_4_weeks"]
PLACERS = {"optimizer": optimize_ambulances, "simple rule": place_ambulances}

fc = pd.read_pickle("data/processed/forecast_2025.pkl")
cent = pd.read_csv("data/processed/fsa_centroids.csv")
cent = cent[cent["fsa"].isin(fc["fsa"].unique())].sort_values("fsa").reset_index(drop=True)
order = cent["fsa"].tolist()
dist = distance_matrix(cent)

# one row per window, areas in the same order as the distance matrix
wide = {col: fc.pivot(index="block", columns="fsa", values=col).reindex(columns=order)
        for col in FORECASTS + ["calls", "urgent_calls"]}
blocks = wide["calls"].index[wide["calls"].notna().any(axis=1)]  # skip outage windows
print(f"Replaying {len(blocks):,} windows x {len(AMBULANCES)} fleet sizes...")

rows = []
start = time.time()
for b_i, block in enumerate(blocks):
    calls = wide["calls"].loc[block].fillna(0).to_numpy()
    urgent = wide["urgent_calls"].loc[block].fillna(0).to_numpy()
    for n in AMBULANCES:
        for p_name, placer in PLACERS.items():
            # each forecast, plus hindsight (planning with the real calls)
            plans = {f: placer(wide[f].loc[block].fillna(0).to_numpy(), dist, n) for f in FORECASTS}
            plans["hindsight"] = placer(calls, dist, n)
            for f, picks in plans.items():
                rows.append({
                    "block": block, "ambulances": n, "placement": p_name, "forecast": f,
                    "calls": calls.sum(),
                    "covered": coverage(calls, dist, picks) * calls.sum(),
                    "urgent": urgent.sum(),
                    "urgent_covered": coverage(urgent, dist, picks) * urgent.sum() if urgent.sum() else 0,
                    "km_x_calls": nearest_km(calls, dist, picks) * calls.sum(),
                })
    if (b_i + 1) % 500 == 0:
        print(f"  {b_i + 1:,} / {len(blocks):,} windows  ({time.time() - start:.0f}s)")

r = pd.DataFrame(rows)
r.to_csv("data/processed/replay_2025.csv", index=False)

# totals over the year: share of all calls reached, not an average of per-window shares
s = r.groupby(["ambulances", "placement", "forecast"])[
    ["calls", "covered", "urgent", "urgent_covered", "km_x_calls"]].sum()
summary = pd.DataFrame({
    "calls reached": s["covered"] / s["calls"],
    "urgent reached": s["urgent_covered"] / s["urgent"],
    "avg km to ambulance": s["km_x_calls"] / s["calls"],
}).reset_index()
summary.to_csv("data/processed/replay_summary.csv", index=False)

print(f"\nDone in {time.time() - start:.0f}s\n")
for n in AMBULANCES:
    print(f"{n} AMBULANCES")
    t = summary[summary["ambulances"] == n].copy()
    t["calls reached"] = t["calls reached"].map("{:.1%}".format)
    t["urgent reached"] = t["urgent reached"].map("{:.1%}".format)
    t["avg km to ambulance"] = t["avg km to ambulance"].map("{:.2f}".format)
    print(t.drop(columns="ambulances").to_string(index=False), "\n")