"""
placement logic for the tool. no streamlit here, so it can be tested on its own.
"""

import numpy as np
import pandas as pd

# an ambulance "covers" an area if the area's centre is within this distance.
# 8 min target x ~30 km/h city driving = 4 km. straight-line for now, roads later.
COVER_KM = 4.0


def distance_matrix(cent):
    # km between every pair of area centres
    xy = cent[["x_km", "y_km"]].to_numpy()
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))


def place_ambulances(demand, dist, n, cover_km=COVER_KM):
    """
    greedy placement: put each ambulance where it covers the most uncovered demand.
    after an area is covered, its remaining demand is halved, so busy areas can
    still attract a second or third ambulance.
    demand: array of expected calls per area. returns list of area indexes.
    """
    covers = dist <= cover_km
    remaining = np.asarray(demand, dtype=float).copy()
    picks = []
    for _ in range(n):
        gain = covers @ remaining          # demand each site would cover
        best = int(np.argmax(gain))
        picks.append(best)
        remaining[covers[best]] *= 0.5
    return picks


def coverage(actual, dist, picks, cover_km=COVER_KM):
    # share of actual calls in areas within reach of at least one ambulance
    actual = np.nan_to_num(np.asarray(actual, dtype=float))
    if actual.sum() == 0 or not picks:
        return float("nan")
    reached = (dist[picks] <= cover_km).any(axis=0)
    return actual[reached].sum() / actual.sum()


def nearest_km(actual, dist, picks):
    # average distance from each actual call's area to its nearest ambulance
    actual = np.nan_to_num(np.asarray(actual, dtype=float))
    if actual.sum() == 0 or not picks:
        return float("nan")
    near = dist[picks].min(axis=0)
    return (near * actual).sum() / actual.sum()


def evaluate_year(fc, cent, method, n, cover_km=COVER_KM):
    """
    replay every 3-hour block of the year: place with the forecast, score on what happened.
    returns average coverage for the forecast plan and the perfect-hindsight plan.
    """
    dist = distance_matrix(cent)
    order = cent["fsa"].tolist()
    res = []
    for block, g in fc.groupby("block"):
        g = g.set_index("fsa").reindex(order)
        if g["calls"].isna().all():
            continue
        plan = place_ambulances(g[method].fillna(0), dist, n, cover_km)
        best = place_ambulances(g["calls"].fillna(0), dist, n, cover_km)
        res.append((coverage(g["calls"], dist, plan, cover_km),
                    coverage(g["calls"], dist, best, cover_km)))
    res = np.array(res)
    return np.nanmean(res[:, 0]), np.nanmean(res[:, 1])