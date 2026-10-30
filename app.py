
import datetime as dt
import json
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
TEXT = "#8b91a3"
WHITE = "#f2f3f5"
LINE = "#25282f"
BOX = "#1c1e23"
ORANGE = "#ef7a45"
FONT = "Roboto Mono, monospace"

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
    "cov": "Percentage of actual calls within 4 km of an ambulance.",
    "best": "Placement using actual calls instead of predicted calls.",
    "map": "Orange shows call volume. White dots show ambulance locations."
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


# info button
def info(key, label):
    return (
        f'<span class="info-wrap">'
        f'<button class="info" type="button" '
        f'aria-label="about {label}" aria-describedby="tip-{key}">i</button>'
        f'<span class="tip" role="tooltip" id="tip-{key}">'
        f'{help_text[key]}</span></span>'
    )


# sidebar
st.sidebar.markdown(
    '<p class="side-title">plan a shift</p>'
    '<p class="side-sub">change anything below and the map updates.</p>',
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
    "ambulances available", 5, 60, 20, help=help_text["count"]
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

# page title
st.markdown(
    f'<p class="eyebrow">{block:%A, %B %d %Y}'.lower()
    + f' · {slot}</p>'
    '<h1 class="title">TPS planner</h1>'
    '<p class="role">where should free ambulances wait, based on where calls are likely to come from.</p>',
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
    st.warning("No records for this time window. Please choose another.")
    st.stop()

# ambulance locations
predicted = data[forecast].fillna(0)
actual = data["calls"].fillna(0)

plan = placer(predicted, dist, n)
best = placer(actual, dist, n)


# stats
def stat(key, label, value, highlight=False):
    style = "value accent" if highlight else "value"

    return (
        f'<div class="stat">'
        f'<div class="label">{label}{info(key, label)}</div>'
        f'<div class="{style}">{value}</div>'
        f'</div>'
    )


st.markdown(
    '<div class="stats">'
    + stat("pred", "predicted calls", f"{predicted.sum():.0f}")
    + stat("actual", "actual calls", f"{actual.sum():.0f}")
    + stat(
        "cov",
        f"calls within {COVER_KM:.0f} km",
        f"{coverage(actual, dist, plan):.0%}",
        True
    )
    + stat(
        "best",
        "hindsight plan",
        f"{coverage(actual, dist, best):.0%}"
    )
    + '</div>'
    + f'<p class="map-head">map{info("map", "the map")}</p>',
    unsafe_allow_html=True
)


# map
colour = forecast if show == "forecast" else "calls"

fig = px.choropleth_map(
    data,
    geojson=shapes,
    locations="fsa",
    color=colour,
    color_continuous_scale=[BOX, "#7a4a35", ORANGE, "#ffc9a8"],
    range_color=(0, max(1, data[colour].max())),
    map_style="carto-darkmatter",
    zoom=9.6,
    center={"lat": 43.71, "lon": -79.38},
    opacity=0.8,
    hover_name="fsa",
    hover_data={"fsa": False, forecast: ":.2f", "calls": ":.0f"},
    labels={forecast: "forecast", "calls": "actual"}
)

fig.update_traces(
    marker_line_color=LINE,
    marker_line_width=0.6,
    selector=dict(type="choroplethmap")
)

# ambulance markers
counts = pd.Series(plan).value_counts()
ambulance_points = cent.iloc[counts.index].copy()
ambulance_points["n"] = counts.values

fig.add_trace(
    go.Scattermap(
        lat=ambulance_points["lat"],
        lon=ambulance_points["lon"],
        mode="markers+text",
        marker=dict(
            size=12 + 4 * ambulance_points["n"],
            color=WHITE
        ),
        text=ambulance_points["n"].where(
            ambulance_points["n"] > 1, ""
        ).astype(str),
        textfont=dict(color=BG, family=FONT, size=13),
        hovertext=[
            f"{fsa}: {count} ambulance(s)"
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
    height=620,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    paper_bgcolor=BG,
    font=dict(family=FONT, color=TEXT, size=15),
    hoverlabel=dict(
        bgcolor=BOX,
        bordercolor=LINE,
        font=dict(family=FONT, color=WHITE, size=15)
    ),
    coloraxis_colorbar=dict(
        title="calls",
        thickness=14,
        len=0.5,
        outlinewidth=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displayModeBar": False}
)

# distance comparison
plan_km = nearest_km(actual, dist, plan)
best_km = nearest_km(actual, dist, best)

st.markdown(
    f'<p class="note">avg distance from a call to the nearest ambulance: '
    f'{plan_km:.1f} km (plan) vs {best_km:.1f} km (hindsight). '
    'straight-line between area centres for now.</p>',
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
    names["hindsight"] = "hindsight (perfect forecast)"

    summary["forecast"] = summary["forecast"].map(names).fillna(
        summary["forecast"]
    )

    summary = summary.sort_values(
        ["placement", "calls reached"],
        ascending=[True, False]
    )

    rows = ""

    for _, row in summary.iterrows():
        rows += (
            f'<tr>'
            f'<td>{row["placement"]}</td>'
            f'<td>{row["forecast"]}</td>'
            f'<td>{row["calls reached"]:.1%}</td>'
            f'<td>{row["urgent reached"]:.1%}</td>'
            f'<td>{row["avg km to ambulance"]:.2f}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<p class="map-head" style="margin-top:40px">'
        f'all of 2025, {closest} ambulances</p>'
        '<p class="role">every 3-hour window replayed: plan with the forecast, score on what really happened.</p>'
        '<table class="yr">'
        '<thead><tr>'
        '<th>placement</th>'
        '<th>forecast</th>'
        '<th>calls reached</th>'
        '<th>urgent reached</th>'
        '<th>avg km</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True
    )

# footer
st.markdown(
    '<p class="footer">© 2026 Amanda Xuan</p>',
    unsafe_allow_html=True
)
