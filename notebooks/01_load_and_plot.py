
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW = Path("data/raw/paramedic-services-incident-data-2021-2025.xlsx")
SAVED = Path("data/processed/incidents_raw.pkl")

# load the data
if SAVED.exists():
    print("Loading saved copy...")
    df = pd.read_pickle(SAVED)
else:
    print("Reading Excel (first time only, this takes a few minutes)...")
    sheets = [pd.read_excel(RAW, sheet_name=str(year)) for year in range(2021, 2026)]
    df = pd.concat(sheets, ignore_index=True)
    SAVED.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(SAVED)

# checking if it looks right 
print(f"\nRows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"From {df['Dispatch_Time'].min()} to {df['Dispatch_Time'].max()}")
print("\nFirst 5 rows:")
print(df.head())

# 1st chart
calls_by_hour = df["Dispatch_Time"].dt.hour.value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(calls_by_hour.index, calls_by_hour.values)
plt.xlabel("Hour of day")
plt.ylabel("Number of calls (2021-2025)")
plt.title("Toronto paramedic calls by hour of day")
plt.xticks(range(24))
plt.tight_layout()

Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/calls_by_hour.png", dpi=150)
print("\nChart saved to figures/calls_by_hour.png")
plt.show()