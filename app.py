import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from planner import (
    COVER_KM, coverage, distance_matrix, nearest_km,
    optimize_ambulances, place_ambulances
)

# colours
BG = "#131417"
CARD = "#191a1e"
LINE = "rgba(237, 238, 241, 0.08)"
TEXT = "#eceef1"
TEXT_2 = "#a4aaba"
TEXT_3 = "#80869a"
ORANGE = "#ef7a45"
MONO = "Roboto Mono, monospace"

st.set_page_config(
    page_title="TPS planner | Amanda Xuan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# load css
with open("style.css") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

# forecast options
methods = {
    "neural network": "nn",
    "machine learning": "gbm",
    "historical average": "hist_avg",
    "last 4 weeks": "last_4_weeks",
    "last week": "last_week"
}

slots = [
    "12am–3am", "3am–6am", "6am–9am", "9am–12pm",
    "12pm–3pm", "3pm–6pm", "6pm–9pm", "9pm–12am"
]

# explanations
help_text = {
    "date": "Choose a day in 2025. The models were trained using earlier data.",
    "slot": "Each day is divided into 3-hour windows.",
    "count": "Number of ambulances available to place.",
    "forecast": "Choose which model predicts the number of calls.",
    "placement": "Optimizer finds a placement that covers the most predicted calls. Simple rule places ambulances one at a time.",
    "colour": "Show forecasted calls or actual calls on the map.",
    "pred": "Total number of predicted calls in this time window.",
    "actual": "Number of calls that actually happened.",
    "cov": "Percentage of actual calls within 4 km of an ambulance. The white tick is the hindsight plan: the same method given the real calls.",
    "map": "Orange shows call volume. White dots are ambulances, and each ring is the 4 km an ambulance can reach.",
    "year": "Every 3-hour window of 2025 replayed: plan with the forecast, then score on what really happened."
}


# load data
@st.cache_data
def load_data():
    forecasts = pd.read_pickle("data/processed/forecast_2025.pkl")
    centres = pd.read_csv("data/processed/fsa_centroids.csv")

    centres = centres[centres["fsa"].isin(forecasts["fsa"].unique())]
    centres = centres.reset_index(drop=True)

    with open("data/processed/toronto_fsa.geojson") as file:
        shapes = json.load(file)

    return forecasts, centres, shapes


fc, cent, shapes = load_data()

# only show forecasts that exist
methods = {
    name: col for name, col in methods.items()
    if col in fc.columns
}

dist = distance_matrix(cent)
areas = cent["fsa"].tolist()


# info button: explanation shows on hover, tap or keyboard focus
def info(key, label):
    return (
        f'<span class="info-wrap">'
        f'<button class="info" type="button" '
        f'aria-label="about {label}" aria-describedby="tip-{key}">i</button>'
        f'<span class="tip" role="tooltip" id="tip-{key}">'
        f'{help_text[key]}</span></span>'
    )


# 4 km reach ring around a point, as map coordinates
def ring(lat, lon, km=COVER_KM, points=48):
    lats, lons = [], []
    for i in range(points + 1):
        angle = 2 * math.pi * i / points
        lats.append(lat + km / 110.57 * math.sin(angle))
        lons.append(lon + km / (111.32 * math.cos(math.radians(43.7))) * math.cos(angle))
    return lats, lons


# sidebar
st.sidebar.markdown(
    '<p class="side-title">plan a shift</p>'
    '<p class="side-sub">change anything and the map updates</p>',
    unsafe_allow_html=True
)

day = st.sidebar.date_input(
    "date (2025)",
    value=dt.date(2025, 8, 1),
    min_value=dt.date(2025, 1, 1),
    max_value=dt.date(2025, 12, 31),
    help=help_text["date"]
)

slot = st.sidebar.selectbox(
    "time window", slots, index=7, help=help_text["slot"]
)

n = st.sidebar.slider(
    "ambulances available", 5, 60, 10, help=help_text["count"]
)

forecast_name = st.sidebar.selectbox(
    "forecast", list(methods), help=help_text["forecast"]
)

forecast = methods[forecast_name]

placement = st.sidebar.selectbox(
    "placement",
    ["optimizer", "simple rule"],
    help=help_text["placement"]
)

if placement == "optimizer":
    placer = optimize_ambulances
else:
    placer = place_ambulances

show = st.sidebar.radio(
    "map colour",
    ["forecast", "what actually happened"],
    help=help_text["colour"]
)


# selected time window
block = pd.Timestamp(day) + pd.Timedelta(hours=3 * slots.index(slot))

data = fc[fc["block"] == block]
data = data.set_index("fsa").reindex(areas).reset_index()

# page title: the window you're planning is the headline
day_name = f"{block:%A, %B} {block.day}".lower()

st.markdown(
    '<p class="eyebrow">TPS planner · toronto paramedic services</p>'
    f'<h1 class="title">{day_name} <span class="title-dim">{slot}</span></h1>'
    '<p class="role">where free ambulances should wait, based on where calls are likely to come from</p>',
    unsafe_allow_html=True
)

with st.expander("how to use this"):
    st.write(
        "1. Choose a date and time window.\n"
        "2. Select how many ambulances are available.\n"
        "3. Choose a forecast and placement method.\n"
        "4. Check the results and map."
    )

# check missing data
if data["calls"].isna().all():
    st.markdown(
        '<p class="warn">no records for this window, the system was down or the clocks changed. '
        'pick another window</p>',
        unsafe_allow_html=True
    )
    st.stop()

# ambulance locations
predicted = data[forecast].fillna(0)
actual = data["calls"].fillna(0)

plan = placer(predicted, dist, n)
best = placer(actual, dist, n)

plan_cov = coverage(actual, dist, plan)
best_cov = coverage(actual, dist, best)
off_by = predicted.sum() - actual.sum()

# gap from the rounded numbers people see, so 77% and 85% always read as 8
gap = round(best_cov * 100) - round(plan_cov * 100)
if gap > 0:
    gap_text = f"hindsight plan {best_cov:.0%}, {gap} points a perfect forecast would add"
else:
    gap_text = f"hindsight plan {best_cov:.0%}, your plan already matches it"


# numbers: two compact figures, then the one that matters, wider
st.markdown(
    '<div class="stats">'

    '<div class="stat">'
    f'<div class="label">predicted calls{info("pred", "predicted calls")}</div>'
    f'<div class="value">{predicted.sum():.0f}</div>'
    f'<div class="meta">{off_by:+.0f} vs actual</div>'
    '</div>'

    '<div class="stat">'
    f'<div class="label">actual calls{info("actual", "actual calls")}</div>'
    f'<div class="value">{actual.sum():.0f}</div>'
    '<div class="meta">2025 records</div>'
    '</div>'

    '<div class="stat key">'
    f'<div class="label"><span class="reach" aria-hidden="true"></span>'
    f'calls within {COVER_KM:.0f} km of an ambulance{info("cov", "calls within 4 km")}</div>'
    '<div class="key-row">'
    f'<div class="value accent">{plan_cov:.0%}</div>'
    '<div class="meter" role="img" '
    f'aria-label="your plan {plan_cov:.0%}, hindsight plan {best_cov:.0%}">'
    f'<span class="fill" style="width:{plan_cov * 100:.1f}%"></span>'
    f'<span class="tick" style="left:{best_cov * 100:.1f}%"></span>'
    '</div>'
    '</div>'
    f'<div class="meta">{gap_text}</div>'
    '</div>'

    '</div>'
    f'<p class="section-head">map{info("map", "the map")}</p>',
    unsafe_allow_html=True
)


# map: graphite tiles, orange heat, white ambulances with their 4 km reach
colour = forecast if show == "forecast" else "calls"

fig = px.choropleth_map(
    data,
    geojson=shapes,
    locations="fsa",
    color=colour,
    color_continuous_scale=["#22242a", "#6b3f2c", ORANGE, "#ffd2b8"],
    range_color=(0, max(1, data[colour].max())),
    map_style="carto-darkmatter",
    zoom=9.6,
    center={"lat": 43.71, "lon": -79.38},
    opacity=0.82,
    hover_name="fsa",
    hover_data={"fsa": False, forecast: ":.2f", "calls": ":.0f"},
    labels={forecast: "forecast", "calls": "actual"}
)

fig.update_traces(
    marker_line_color="rgba(237, 238, 241, 0.10)",
    marker_line_width=0.6,
    selector=dict(type="choroplethmap")
)

# ambulance markers, sized by how many share a spot
counts = pd.Series(plan).value_counts()
ambulance_points = cent.iloc[counts.index].copy()
ambulance_points["n"] = counts.values

# reach rings, drawn as one trace
ring_lats, ring_lons = [], []
for lat, lon in zip(ambulance_points["lat"], ambulance_points["lon"]):
    lats, lons = ring(lat, lon)
    ring_lats += lats + [None]
    ring_lons += lons + [None]

fig.add_trace(
    go.Scattermap(
        lat=ring_lats,
        lon=ring_lons,
        mode="lines",
        line=dict(color="rgba(237, 238, 241, 0.45)", width=1),
        hoverinfo="skip",
        name="4 km reach"
    )
)

fig.add_trace(
    go.Scattermap(
        lat=ambulance_points["lat"],
        lon=ambulance_points["lon"],
        mode="markers+text",
        marker=dict(
            size=12 + 4 * ambulance_points["n"],
            color=TEXT
        ),
        text=ambulance_points["n"].where(
            ambulance_points["n"] > 1, ""
        ).astype(str),
        textfont=dict(color=BG, family=MONO, size=12),
        hovertext=[
            f"{fsa}: {count} ambulance{'s' if count > 1 else ''}"
            for fsa, count in zip(
                ambulance_points["fsa"],
                ambulance_points["n"]
            )
        ],
        hoverinfo="text",
        name="ambulances"
    )
)

fig.update_layout(
    height=560,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    paper_bgcolor=BG,
    font=dict(family=MONO, color=TEXT_2, size=13),
    hoverlabel=dict(
        bgcolor=CARD,
        bordercolor="rgba(237, 238, 241, 0.16)",
        font=dict(family=MONO, color=TEXT, size=14)
    ),
    # slim legend inside the map, no empty band beside it
    coloraxis_colorbar=dict(
        title=dict(text="calls", font=dict(color=TEXT_2, size=12)),
        tickfont=dict(color=TEXT_2, size=12),
        thickness=8,
        len=0.4,
        x=0.985,
        xanchor="right",
        y=0.96,
        yanchor="top",
        outlinewidth=0,
        bgcolor="rgba(19, 20, 23, 0.72)"
    )
)

st.plotly_chart(
    fig,
    width="stretch",
    config={"displayModeBar": False}
)

# distance comparison
plan_km = nearest_km(actual, dist, plan)
best_km = nearest_km(actual, dist, best)

st.markdown(
    f'<p class="note">nearest ambulance on average: '
    f'<span class="num">{plan_km:.1f} km</span> plan · '
    f'<span class="num">{best_km:.1f} km</span> hindsight · '
    'straight-line between area centres</p>',
    unsafe_allow_html=True
)


# yearly results
summary_path = Path("data/processed/replay_summary.csv")

if summary_path.exists():
    summary = pd.read_csv(summary_path)

    sizes = summary["ambulances"].unique()
    closest = sizes[abs(sizes - n).argmin()]

    summary = summary[summary["ambulances"] == closest].copy()

    names = {value: key for key, value in methods.items()}
    names["hindsight"] = "perfect forecast"

    summary["name"] = summary["forecast"].map(names).fillna(summary["forecast"])
    summary = summary.sort_values("calls reached", ascending=False)

    real = summary[summary["forecast"] != "hindsight"]
    top = real.index[0] if len(real) else None

    rows = ""

    for i, row in summary.iterrows():
        kind = "ref" if row["forecast"] == "hindsight" else ("top" if i == top else "")
        rows += (
            f'<li class="yr-row {kind}">'
            f'<span class="yr-name">{row["placement"]} · {row["name"]}</span>'
            f'<span class="yr-bar"><span style="width:{row["calls reached"] * 100:.1f}%"></span></span>'
            f'<span class="yr-num">{row["calls reached"]:.1%}</span>'
            f'<span class="yr-meta">urgent {row["urgent reached"]:.1%} · {row["avg km to ambulance"]:.2f} km</span>'
            f'</li>'
        )

    st.markdown(
        f'<p class="section-head yr-head">all of 2025 with {closest} ambulances'
        f'{info("year", "all of 2025")}</p>'
        '<p class="role">share of real calls reached within 4 km, every 3-hour window replayed</p>'
        f'<ul class="yr">{rows}</ul>',
        unsafe_allow_html=True
    )

# footer
st.markdown(
    '<p class="footer">© 2026 Amanda Xuan · data from city of toronto open data and statistics canada</p>',
    unsafe_allow_html=True
)