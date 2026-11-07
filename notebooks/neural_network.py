
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

torch.manual_seed(0)
np.random.seed(0)

# load data
t = pd.read_pickle("data/processed/zone_blocks.pkl")
t = t.sort_values(["fsa", "block"]).reset_index(drop=True)

week = 7 * 8
month = 4 * week

# time and location
t["slot"] = t["block"].dt.hour // 3
t["dow"] = t["block"].dt.dayofweek
t["month"] = t["block"].dt.month - 1
t["fsa_code"] = t["fsa"].astype("category").cat.codes

# previous call counts
area = t.groupby("fsa")["calls"]

t["last_week"] = area.shift(week)
t["two_weeks_ago"] = area.shift(2 * week)

t["last_4_weeks"] = pd.concat(
    [area.shift(week * i) for i in range(1, 5)], axis=1
).mean(axis=1)

t["area_recent"] = area.transform(
    lambda x: x.shift(week).rolling(month, min_periods=week).mean()
)

city = t.groupby("block")["calls"].sum(min_count=1)
city_avg = city.shift(week).rolling(month, min_periods=week).mean()
t["city_recent"] = t["block"].map(city_avg)

# split by year
year = t["block"].dt.year

train = (year <= 2023) & t["calls"].notna() & t["last_4_weeks"].notna()
val = (year == 2024) & t["calls"].notna()
test = year == 2025

# average calls for each area and time
avg = t[train].groupby(["fsa", "dow", "slot"])["calls"].mean()
t = t.join(avg.rename("hist_avg"), on=["fsa", "dow", "slot"])

numeric = [
    "last_week", "two_weeks_ago", "last_4_weeks",
    "area_recent", "city_recent", "hist_avg"
]

categories = {
    "fsa_code": int(t["fsa_code"].max() + 1),
    "slot": 8,
    "dow": 7,
    "month": 12
}

# scale numbers
num = np.log1p(t[numeric].astype(float))
mean = num[train].mean()
std = num[train].std()

num = ((num - mean) / std).fillna(0)


def make_tensors(mask):
    x1 = torch.tensor(num[mask].to_numpy(), dtype=torch.float32)
    x2 = torch.tensor(
        t.loc[mask, list(categories)].to_numpy(), dtype=torch.long
    )
    y = torch.tensor(
        t.loc[mask, "calls"].fillna(0).to_numpy(), dtype=torch.float32
    )
    return x1, x2, y


train_data = make_tensors(train)
val_data = make_tensors(val)
test_data = make_tensors(test)

print("Train:", len(train_data[2]))
print("Validate:", len(val_data[2]))
print("Test:", len(test_data[2]))


# neural network
class DemandNet(nn.Module):

    def __init__(self):
        super().__init__()

        sizes = {"fsa_code": 8, "slot": 4, "dow": 3, "month": 3}

        self.embeds = nn.ModuleList([
            nn.Embedding(categories[name], sizes[name])
            for name in categories
        ])

        inputs = len(numeric) + sum(sizes.values())

        self.layers = nn.Sequential(
            nn.Linear(inputs, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x1, x2):
        parts = []

        for i, emb in enumerate(self.embeds):
            parts.append(emb(x2[:, i]))

        parts.append(x1)
        x = torch.cat(parts, dim=1)

        return self.layers(x).squeeze(1)


model = DemandNet()

# start near the average number of calls
with torch.no_grad():
    model.layers[-1].bias.fill_(
        float(np.log(train_data[2].mean().item()))
    )

loss_fn = nn.PoissonNLLLoss(log_input=True, full=False)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.002, weight_decay=0.0001
)

batch_size = 2048
max_epochs = 25
patience = 3


# check model loss
def evaluate(data):
    model.eval()
    x1, x2, y = data
    total = 0

    with torch.no_grad():
        for i in range(0, len(y), 65536):
            end = i + 65536

            pred = model(x1[i:end], x2[i:end])
            loss = loss_fn(pred, y[i:end])

            total += loss.item() * len(y[i:end])

    return total / len(y)


# train model
best_loss = float("inf")
best_model = None
bad_epochs = 0
start = time.time()

for epoch in range(1, max_epochs + 1):
    model.train()

    x1, x2, y = train_data
    order = torch.randperm(len(y))

    for i in range(0, len(y), batch_size):
        idx = order[i:i + batch_size]

        optimizer.zero_grad()
        pred = model(x1[idx], x2[idx])
        loss = loss_fn(pred, y[idx])

        loss.backward()
        optimizer.step()

    train_loss = evaluate(train_data)
    val_loss = evaluate(val_data)

    print(
        f"Epoch {epoch}: train={train_loss:.4f}, "
        f"val={val_loss:.4f}, time={time.time() - start:.0f}s"
    )

    # stop if validation is not improving
    if val_loss < best_loss - 0.0001:
        best_loss = val_loss
        bad_epochs = 0
        best_model = {
            k: v.clone() for k, v in model.state_dict().items()
        }
    else:
        bad_epochs += 1

        if bad_epochs >= patience:
            print("Stopping early")
            break


# save model
model.load_state_dict(best_model)
torch.save(best_model, "data/processed/nn_model.pt")


# predict 2025
model.eval()

with torch.no_grad():
    x1, x2, y = test_data
    predictions = torch.exp(model(x1, x2)).numpy()

t.loc[test, "nn"] = predictions


# compare predictions
fc = pd.read_pickle("data/processed/forecast_2025.pkl")
fc = fc.drop(columns="nn", errors="ignore")

fc = fc.merge(
    t.loc[test, ["fsa", "block", "nn"]],
    on=["fsa", "block"]
)

methods = ["last_week", "last_4_weeks", "hist_avg", "gbm", "nn"]

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


# save predictions
fc.to_pickle("data/processed/forecast_2025.pkl")

print("\nModel and predictions saved")
