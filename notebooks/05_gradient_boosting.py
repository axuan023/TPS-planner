
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# load data
t = pd.read_pickle("data/processed/zone_blocks.pkl")
t = t.sort_values(["fsa", "block"]).reset_index(drop=True)

week = 7 * 8
month = 4 * week

# time and location
t["slot"] = t["block"].dt.hour // 3
t["dow"] = t["block"].dt.dayofweek
t["month"] = t["block"].dt.month
t["fsa_code"] = t["fsa"].astype("category").cat.codes

# previous calls in each area
area = t.groupby("fsa")["calls"]

t["last_week"] = area.shift(week)
t["two_weeks_ago"] = area.shift(2 * week)

past_weeks = []

for i in range(1, 5):
    past_weeks.append(area.shift(week * i))

t["last_4_weeks"] = pd.concat(past_weeks, axis=1).mean(axis=1)

# average calls in the last few weeks
t["area_recent"] = area.transform(
    lambda x: x.shift(week).rolling(month, min_periods=week).mean()
)

# recent calls across the city
city = t.groupby("block")["calls"].sum(min_count=1)
city_avg = city.shift(week).rolling(month, min_periods=week).mean()

t["city_recent"] = t["block"].map(city_avg)

# training and testing data
train = (
    (t["block"] < "2025-01-01")
    & t["calls"].notna()
    & t["last_4_weeks"].notna()
)

test = t["block"] >= "2025-01-01"

# average calls for each area and time
avg = t[train].groupby(["fsa", "dow", "slot"])["calls"].mean()
t = t.join(avg.rename("hist_avg"), on=["fsa", "dow", "slot"])

features = [
    "fsa_code",
    "slot",
    "dow",
    "month",
    "last_week",
    "two_weeks_ago",
    "last_4_weeks",
    "area_recent",
    "city_recent",
    "hist_avg"
]

# train gradient boosting model
model = HistGradientBoostingRegressor(
    loss="poisson",
    learning_rate=0.05,
    max_iter=400,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    categorical_features=["fsa_code"],
    early_stopping=True,
    validation_fraction=0.1,
    random_state=0
)

print(f"Training on {train.sum()} rows...")

start = time.time()

model.fit(t.loc[train, features], t.loc[train, "calls"])

print(f"Training finished in {time.time() - start:.0f} seconds")
print(f"Trees used: {model.n_iter_}")

# predict calls for 2025
t.loc[test, "gbm"] = model.predict(t.loc[test, features])

# load previous forecasts
fc = pd.read_pickle("data/processed/forecast_baseline_2025.pkl")

fc = fc.merge(
    t.loc[test, ["fsa", "block", "gbm"]],
    on=["fsa", "block"]
)

# compare forecast errors
methods = ["last_week", "last_4_weeks", "hist_avg", "gbm"]

results = fc[fc["calls"].notna()].dropna(subset=methods)
city_actual = results.groupby("block")["calls"].sum()

print("\n2025 RESULTS\n")
print(f"{'Method':<15}{'Area Error':>12}{'City Error':>12}{'City Bias':>12}")

for method in methods:
    area_error = np.mean(abs(results[method] - results["calls"]))

    city_pred = results.groupby("block")[method].sum()
    city_error = np.mean(abs(city_pred - city_actual))
    city_bias = (city_pred - city_actual).mean()

    print(
        f"{method:<15}{area_error:>12.3f}"
        f"{city_error:>12.1f}{city_bias:>+12.1f}"
    )

# save results
fc.to_pickle("data/processed/forecast_2025.pkl")

print("\nForecast saved")
