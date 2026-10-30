"""
step 2: clean the data and build the zone-time table.

run: python notebooks/02_clean.py
in:  data/processed/incidents_raw.pkl
out: data/processed/incidents_clean.pkl, data/processed/zone_blocks.pkl
"""

from pathlib import Path

import pandas as pd

df = pd.read_pickle("data/processed/incidents_raw.pkl")
print(f"Starting rows: {len(df):,}")


def report(step, before, after):
    # print how many rows each step removed
    print(f"  {step:<45} removed {before - after:>7,}   ({after:,} left)")


# drop the duplicate incident id
n = len(df)
df = df.drop_duplicates(subset="ID", keep="first")
report("Duplicate IDs", n, len(df))

# keep toronto areas only (M + digit + letter)
n = len(df)
df = df[df["Forward_Sortation_Area"].str.match(r"^M\d[A-Z]$")]
report("Outside Toronto or unknown location", n, len(df))

# drop non-emergency types
n = len(df)
df = df[~df["Incident_Type"].isin(["Airport Standby", "Administrative"])]
report("Airport standby / administrative", n, len(df))

# urgent = echo (9) or delta (1)
df["urgent"] = df["Priority_Number"].isin([1, 9])
df = df.rename(columns={"Forward_Sortation_Area": "fsa"})

# hours with <5 calls city-wide are recording gaps (outages, daylight saving)
hourly = df.set_index("Dispatch_Time").resample("h").size()
bad_hours = hourly[hourly < 5].index
print(f"\nHours with missing records: {len(bad_hours)}")
for h in bad_hours:
    print(f"  {h}  ({hourly[h]} calls)")

df.to_pickle("data/processed/incidents_clean.pkl")
print(f"\nClean calls saved: {len(df):,}")

# count calls per area per 3-hour block
BLOCK = "3h"
df["block"] = df["Dispatch_Time"].dt.floor(BLOCK)
counts = df.groupby(["fsa", "block"]).agg(
    calls=("ID", "size"), urgent_calls=("urgent", "sum")
)

# include every area x block, with 0 where nothing happened
all_fsas = sorted(df["fsa"].unique())
all_blocks = pd.date_range(df["block"].min(), df["block"].max(), freq=BLOCK)
full_index = pd.MultiIndex.from_product([all_fsas, all_blocks], names=["fsa", "block"])
table = counts.reindex(full_index, fill_value=0).reset_index()

# blocks touching a gap get NaN, not a fake count
bad_blocks = set(pd.DatetimeIndex(bad_hours).floor(BLOCK))
table[["calls", "urgent_calls"]] = table[["calls", "urgent_calls"]].astype("float")
table.loc[table["block"].isin(bad_blocks), ["calls", "urgent_calls"]] = float("nan")

table.to_pickle("data/processed/zone_blocks.pkl")

print("\nZONE-TIME TABLE")
print(f"  Areas: {len(all_fsas)}   3-hour blocks: {len(all_blocks):,}")
print(f"  Rows: {len(table):,}")
print(f"  Blocks marked missing: {len(bad_blocks)}")
print(f"  Average calls per area per block: {table['calls'].mean():.2f}")
print(f"  Share of rows with zero calls: {(table['calls'] == 0).mean():.0%}")
print("\nSample rows:")
print(table.sample(8, random_state=1).sort_values(["fsa", "block"]).to_string(index=False))