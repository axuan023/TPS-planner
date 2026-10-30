
import pandas as pd

# load data
df = pd.read_pickle("data/processed/incidents_raw.pkl")

print(f"Starting rows: {len(df):,}")


# show how many rows were removed
def report(step, before):
    removed = before - len(df)
    print(f"{step}: removed {removed:,}, {len(df):,} left")


# remove duplicates
before = len(df)
df = df.drop_duplicates(subset="ID", keep="first")
report("Duplicate IDs", before)

# keep Toronto postal areas
before = len(df)
df = df[df["Forward_Sortation_Area"].str.match(r"^M\d[A-Z]$", na=False)]
report("Invalid locations", before)

# remove non-emergency calls
before = len(df)
df = df[~df["Incident_Type"].isin(["Airport Standby", "Administrative"])]
report("Non-emergency calls", before)

# mark urgent calls
df["urgent"] = df["Priority_Number"].isin([1, 9])

df = df.rename(columns={"Forward_Sortation_Area": "fsa"})

# find hours with missing data
hourly = df.set_index("Dispatch_Time").resample("h").size()

bad_hours = hourly[hourly < 5].index

print(f"\nHours with missing records: {len(bad_hours)}")

for hour in bad_hours:
    print(f"{hour}: {hourly[hour]} calls")

# save cleaned data
df.to_pickle("data/processed/incidents_clean.pkl")

print(f"\nClean calls saved: {len(df):,}")


# group calls into 3-hour blocks
block_size = "3h"

df["block"] = df["Dispatch_Time"].dt.floor(block_size)

counts = df.groupby(["fsa", "block"]).agg(
    calls=("ID", "size"),
    urgent_calls=("urgent", "sum")
)

# create all areas and time blocks
areas = sorted(df["fsa"].unique())

blocks = pd.date_range(
    df["block"].min(),
    df["block"].max(),
    freq=block_size
)

full_index = pd.MultiIndex.from_product(
    [areas, blocks],
    names=["fsa", "block"]
)

# fill missing combinations with zero
table = counts.reindex(full_index, fill_value=0).reset_index()

# mark blocks with missing records
bad_blocks = set(bad_hours.floor(block_size))

table["calls"] = table["calls"].astype(float)
table["urgent_calls"] = table["urgent_calls"].astype(float)

table.loc[
    table["block"].isin(bad_blocks),
    ["calls", "urgent_calls"]
] = float("nan")

# save table
table.to_pickle("data/processed/zone_blocks.pkl")


# check results
print("\nZONE-TIME TABLE")

print(f"Areas: {len(areas)}")
print(f"3-hour blocks: {len(blocks):,}")
print(f"Total rows: {len(table):,}")
print(f"Missing blocks: {len(bad_blocks)}")

print(f"Average calls: {table['calls'].mean():.2f}")
print(f"Zero-call rows: {(table['calls'] == 0).mean():.0%}")

print("\nSample rows:")

sample = table.sample(8, random_state=1)
sample = sample.sort_values(["fsa", "block"])

print(sample.to_string(index=False))
