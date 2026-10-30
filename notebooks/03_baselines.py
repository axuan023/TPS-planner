"""
simple forecasts (baselines) and how to score them.

run: python notebooks/03_baselines.py
in:  data/processed/zone_blocks.pkl
out: data/processed/forecast_baseline_2025.pkl

each method gets to learn and train on 2021-2024, and test on 2025. never let the model see the test year.
"""

import numpy as np
import pandas as pd

t = pd.read_pickle("data/processed/zone_blocks.pkl")
t = t.sort_values(["fsa", "block"]).reset_index(drop=True)

# time features
t["dow"] = t["block"].dt.dayofweek          # 0 = monday
t["slot"] = t["block"].dt.hour // 3         # 0 = midnight-3am ... 7 = 9pm-midnight
train = t["block"] < "2025-01-01"
test = ~train

BLOCKS_PER_WEEK = 7 * 8

# baseline 1: same block last week
t["last_week"] = t.groupby("fsa")["calls"].shift(BLOCKS_PER_WEEK)

# baseline 2: average of same block over the last 4 weeks (skips missing weeks)
lags = [t.groupby("fsa")["calls"].shift(BLOCKS_PER_WEEK * k) for k in range(1, 5)]
t["last_4_weeks"] = pd.concat(lags, axis=1).mean(axis=1)

# baseline 3: historical average for this area, weekday and time slot (train years only)
hist = t[train].groupby(["fsa", "dow", "slot"])["calls"].mean().rename("hist_avg")
t = t.join(hist, on=["fsa", "dow", "slot"])

# score on 2025, skipping missing blocks
ev = t[test & t["calls"].notna()].dropna(subset=["last_week", "last_4_weeks"])
methods = ["last_week", "last_4_weeks", "hist_avg"]


def mae(pred, actual):
    # average size of the error, in calls
    return np.mean(np.abs(pred - actual))


print("SCORES ON 2025 (lower is better)\n")
print(f"{'method':<15}{'area error':>12}{'city error':>12}{'city bias':>12}")
city_actual = ev.groupby("block")["calls"].sum()
for m in methods:
    area_err = mae(ev[m], ev["calls"])
    city_pred = ev.groupby("block")[m].sum()
    city_err = mae(city_pred, city_actual)
    bias = (city_pred - city_actual).mean()
    print(f"{m:<15}{area_err:>12.3f}{city_err:>12.1f}{bias:>+12.1f}")

print("\narea error = avg miss per area per 3h block")
print("city error = avg miss on the city-wide total per 3h block")
print(f"city bias  = avg over(+)/under(-) prediction (actual avg: {city_actual.mean():.1f} calls per block)")

# save 2025 forecasts for the tool
out = t[test][["fsa", "block", "calls", "urgent_calls"] + methods]
out.to_pickle("data/processed/forecast_baseline_2025.pkl")
print("\nSaved: data/processed/forecast_baseline_2025.pkl")

# one example night
ex = out[out["block"] == "2025-08-01 21:00"].sort_values("calls", ascending=False).head(8)
print("\nExample: Friday Aug 1 2025, 9pm-midnight, busiest areas")
print(ex[["fsa", "calls"] + methods].round(2).to_string(index=False))