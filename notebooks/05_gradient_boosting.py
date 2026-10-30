"""
step 5: machine learning forecast (gradient boosting).

run: .venv/Scripts/python notebooks/05_gradient_boosting.py
in:  data/processed/zone_blocks.pkl, data/processed/forecast_baseline_2025.pkl
out: data/processed/forecast_2025.pkl (baselines + the new model)

same setup as step 3: train on 2021-2024, test on 2025.
every feature only uses data from at least a week before, so the model
could really make this forecast a week ahead.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

t = pd.read_pickle("data/processed/zone_blocks.pkl")
t = t.sort_values(["fsa", "block"]).reset_index(drop=True)

WEEK = 7 * 8      # 3-hour blocks in a week
MONTH = 4 * WEEK  # 4 weeks

# calendar features
t["slot"] = t["block"].dt.hour // 3
t["dow"] = t["block"].dt.dayofweek
t["month"] = t["block"].dt.month
t["fsa_code"] = t["fsa"].astype("category").cat.codes

# area history, all shifted back at least one week
by_area = t.groupby("fsa")["calls"]
t["last_week"] = by_area.shift(WEEK)
t["two_weeks_ago"] = by_area.shift(2 * WEEK)
t["last_4_weeks"] = pd.concat([by_area.shift(WEEK * k) for k in range(1, 5)], axis=1).mean(axis=1)

# how busy this area has been overall lately (avg per block over the past 4 weeks)
t["area_recent"] = by_area.transform(lambda s: s.shift(WEEK).rolling(MONTH, min_periods=WEEK).mean())

# how busy the whole city has been lately (picks up growth year over year)
city = t.groupby("block")["calls"].sum(min_count=1)
city_recent = city.shift(WEEK).rolling(MONTH, min_periods=WEEK).mean()
t["city_recent"] = t["block"].map(city_recent)

train = (t["block"] < "2025-01-01") & t["calls"].notna() & t["last_4_weeks"].notna()
test = t["block"] >= "2025-01-01"

# long-run average for this area / weekday / slot, from training years only
hist = t[train].groupby(["fsa", "dow", "slot"])["calls"].mean().rename("hist_avg")
t = t.join(hist, on=["fsa", "dow", "slot"])

FEATURES = ["fsa_code", "slot", "dow", "month", "last_week", "two_weeks_ago",
            "last_4_weeks", "area_recent", "city_recent", "hist_avg"]

# poisson loss: right choice for counts (0, 1, 2 ...), never predicts below 0
model = HistGradientBoostingRegressor(
    loss="poisson",
    learning_rate=0.05,
    max_iter=400,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    categorical_features=["fsa_code"],
    early_stopping=True,
    validation_fraction=0.1,
    random_state=0,
)

print(f"Training on {train.sum():,} rows...")
start = time.time()
model.fit(t.loc[train, FEATURES], t.loc[train, "calls"])
print(f"Done in {time.time() - start:.0f}s, {model.n_iter_} trees")

t.loc[test, "gbm"] = model.predict(t.loc[test, FEATURES])

# score against the baselines from step 3, on the exact same rows
fc = pd.read_pickle("data/processed/forecast_baseline_2025.pkl")
fc = fc.merge(t.loc[test, ["fsa", "block", "gbm"]], on=["fsa", "block"])
methods = ["last_week", "last_4_weeks", "hist_avg", "gbm"]
ev = fc[fc["calls"].notna()].dropna(subset=methods)

print("\nSCORES ON 2025 (lower is better)\n")
print(f"{'method':<15}{'area error':>12}{'city error':>12}{'city bias':>12}")
city_actual = ev.groupby("block")["calls"].sum()
for m in methods:
    area_err = np.mean(np.abs(ev[m] - ev["calls"]))
    city_pred = ev.groupby("block")[m].sum()
    city_err = np.mean(np.abs(city_pred - city_actual))
    bias = (city_pred - city_actual).mean()
    print(f"{m:<15}{area_err:>12.3f}{city_err:>12.1f}{bias:>+12.1f}")

fc.to_pickle("data/processed/forecast_2025.pkl")
print("\nSaved: data/processed/forecast_2025.pkl")