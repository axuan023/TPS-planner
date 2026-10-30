# ambulance planner (version 1).

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from planner import (COVER_KM, coverage, distance_matrix, nearest_km,
                     optimize_ambulances, place_ambulances)

# site colours
BG, TEXT, STRONG, FAINT, LINE, BOX, ACCENT = (
    "#131417", "#8b91a3", "#f2f3f5", "#4a4f5c", "#25282f", "#1c1e23", "#ef7a45")
FONT = "Roboto Mono, monospace"

st.set_page_config(page_title="TPS planner | Amanda Xuan", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{Path('style.css').read_text()}</style>", unsafe_allow_html=True)

METHODS = {
    "neural network": "nn",
    "machine learning": "gbm",
    "historical average": "hist_avg",
    "last 4 weeks": "last_4_weeks",
    "last week": "last_week",
}
SLOTS = ["12am–3am", "3am–6am", "6am–9am", "9am–12pm",
         "12pm–3pm", "3pm–6pm", "6pm–9pm", "9pm–12am"]

# every explanation in one place, used by the info buttons and help icons
HELP = {
    "date": "pick any day in 2025. the model was trained on 2021–2024, so 2025 is a fair test it never saw.",
    "slot": "the day is split into 3-hour windows. forecasts and plans are made for one window at a time.",
    "count": "how many free ambulances you have to position. fewer ambulances makes the placement choice matter more.",
    "forecast": "neural network: a pytorch model that learns a vector for each area and time. "
                "machine learning: gradient boosting trained on area, time and recent calls. "
                "historical average: the usual count for this area, weekday and time. "
                "last 4 weeks / last week: the same window in recent weeks.",
    "placement": "optimizer: solves a covering model that reaches the most predicted calls with your ambulances. "
                 "simple rule: places one ambulance at a time where it helps most, never revisiting earlier picks.",
    "colour": "colour the map by the forecast (what the plan was based on) or by the calls that really happened.",
    "pred": "total calls the chosen forecast expected across toronto in this window.",
    "actual": "total calls that really happened in this window (2025 records).",
    "cov": "share of the real calls that happened within 4 km of one of your ambulances. higher is better.",
    "best": "the same placement method, but given the real calls instead of a forecast. "
            "the gap between this and your plan shows how much a perfect forecast would add.",
    "map": "each area is a postal code region (like M5V). darker orange = more calls. white dots = ambulances; "
           "a number means several wait in that area. hover an area to see its numbers.",
}


@st.cache_data
def load():
    # cached so it only loads once
    fc = pd.read_pickle("data/processed/forecast_2025.pkl")
    cent = pd.read_csv("data/processed/fsa_centroids.csv")
    cent = cent[cent["fsa"].isin(fc["fsa"].unique())].reset_index(drop=True)
    with open("data/processed/toronto_fsa.geojson") as fh:
        gj = json.load(fh)
    return fc, cent, gj


fc, cent, gj = load()
METHODS = {k: v for k, v in METHODS.items() if v in fc.columns}  # only models you've trained
dist = distance_matrix(cent)
order = cent["fsa"].tolist()


def info(key, label):
    # an "i" button that shows its explanation on hover, tap or keyboard focus
    return (f'<span class="info-wrap"><button class="info" type="button" '
            f'aria-label="about {label}" aria-describedby="tip-{key}">i</button>'
            f'<span class="tip" role="tooltip" id="tip-{key}">{HELP[key]}</span></span>')


# sidebar controls
st.sidebar.markdown('<p class="side-title">plan a shift</p>'
                    '<p class="side-sub">change anything below and the map updates.</p>',
                    unsafe_allow_html=True)
day = st.sidebar.date_input("date (2025)", value=dt.date(2025, 8, 1),
                            min_value=dt.date(2025, 1, 1),
                            max_value=dt.date(2025, 12, 31), help=HELP["date"])
slot = st.sidebar.selectbox("time window", SLOTS, index=7, help=HELP["slot"])
n = st.sidebar.slider("ambulances available", 5, 60, 20, help=HELP["count"])
method_label = st.sidebar.selectbox("forecast", list(METHODS), help=HELP["forecast"])
method = METHODS[method_label]
placer_label = st.sidebar.selectbox("placement", ["optimizer", "simple rule"], help=HELP["placement"])
placer = optimize_ambulances if placer_label == "optimizer" else place_ambulances
show = st.sidebar.radio("map colour", ["forecast", "what actually happened"], help=HELP["colour"])

# pick the 3-hour block
block = pd.Timestamp(day) + pd.Timedelta(hours=3 * SLOTS.index(slot))
g = fc[fc["block"] == block].set_index("fsa").reindex(order).reset_index()

# header, same layout as the site
st.markdown(
    f'<p class="eyebrow">{block:%A, %B %d %Y}'.lower() + f' · {slot}</p>'
    '<h1 class="title">TPS planner</h1>'
    '<p class="role">where should free ambulances wait, based on where calls are likely to come from.</p>',
    unsafe_allow_html=True,
)

with st.expander("how to use this", expanded=False):
    st.markdown(
        "1. pick a **date** and **time window** in the sidebar.\n"
        "2. set how many **ambulances** are free.\n"
        "3. choose a **forecast** and a **placement** method.\n"
        "4. read the numbers below and check the map. every **i** button explains what you're looking at."
    )

if g["calls"].isna().all():
    st.markdown('<p class="warn">no records for this window (system outage or daylight saving). '
                'pick another.</p>', unsafe_allow_html=True)
    st.stop()

# place ambulances using the forecast, and with hindsight for comparison
plan = placer(g[method].fillna(0), dist, n)
best = placer(g["calls"].fillna(0), dist, n)


def stat(key, label, value, accent=False):
    cls = "value accent" if accent else "value"
    return (f'<div class="stat"><div class="label">{label}{info(key, label)}</div>'
            f'<div class="{cls}">{value}</div></div>')


st.markdown(
    '<div class="stats">'
    + stat("pred", "predicted calls", f"{g[method].sum():.0f}")
    + stat("actual", "actual calls", f"{g['calls'].sum():.0f}")
    + stat("cov", f"calls within {COVER_KM:.0f} km", f"{coverage(g['calls'], dist, plan):.0%}", accent=True)
    + stat("best", "hindsight plan", f"{coverage(g['calls'], dist, best):.0%}")
    + "</div>"
    + f'<p class="map-head">map{info("map", "the map")}</p>',
    unsafe_allow_html=True,
)

# map: dark tiles, koi-orange heat, white ambulances
col = method if show == "forecast" else "calls"
fig = px.choropleth_map(
    g, geojson=gj, locations="fsa", color=col,
    color_continuous_scale=[BOX, "#7a4a35", ACCENT, "#ffc9a8"],
    range_color=(0, max(1, g[col].max())),
    map_style="carto-darkmatter", zoom=9.6, center={"lat": 43.71, "lon": -79.38},
    opacity=0.8, hover_name="fsa",
    hover_data={"fsa": False, method: ":.2f", "calls": ":.0f"},
    labels={method: "forecast", "calls": "actual"},
)
fig.update_traces(marker_line_color=LINE, marker_line_width=0.6, selector=dict(type="choroplethmap"))

# ambulance markers, sized by how many share a spot
counts = pd.Series(plan).value_counts()
amb = cent.iloc[counts.index].assign(n=counts.values)
fig.add_trace(go.Scattermap(
    lat=amb["lat"], lon=amb["lon"], mode="markers+text",
    marker=dict(size=12 + 4 * amb["n"], color=STRONG),
    text=amb["n"].where(amb["n"] > 1, "").astype(str),
    textfont=dict(color=BG, family=FONT, size=13),
    hovertext=[f"{f}: {k} ambulance(s)" for f, k in zip(amb["fsa"], amb["n"])],
    hoverinfo="text", name="ambulances",
))
fig.update_layout(
    height=620, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
    paper_bgcolor=BG, font=dict(family=FONT, color=TEXT, size=15),
    hoverlabel=dict(bgcolor=BOX, bordercolor=LINE, font=dict(family=FONT, color=STRONG, size=15)),
    coloraxis_colorbar=dict(title="calls", thickness=14, len=0.5, outlinewidth=0,
                            tickfont=dict(color=TEXT, size=14), title_font=dict(color=TEXT, size=14)),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    f'<p class="note">avg distance from a call to the nearest ambulance: '
    f'{nearest_km(g["calls"], dist, plan):.1f} km (plan) vs '
    f'{nearest_km(g["calls"], dist, best):.1f} km (hindsight). '
    'straight-line between area centres for now.</p>',
    unsafe_allow_html=True,
)

# year in review, from notebooks/07_replay.py
summary_path = Path("data/processed/replay_summary.csv")
if summary_path.exists():
    yr = pd.read_csv(summary_path)
    # show the replayed fleet size closest to the slider
    sizes = yr["ambulances"].unique()
    yr = yr[yr["ambulances"] == sizes[abs(sizes - n).argmin()]]
    names = {v: k for k, v in METHODS.items()} | {"hindsight": "hindsight (perfect forecast)"}
    yr = yr.assign(forecast=yr["forecast"].map(names).fillna(yr["forecast"]))
    yr = yr.sort_values(["placement", "calls reached"], ascending=[True, False])
    rows = "".join(
        f'<tr><td>{r.placement}</td><td>{r.forecast}</td><td>{r["calls reached"]:.1%}</td>'
        f'<td>{r["urgent reached"]:.1%}</td><td>{r["avg km to ambulance"]:.2f}</td></tr>'
        for _, r in yr.iterrows())
    st.markdown(
        f'<p class="map-head" style="margin-top:40px">all of 2025, {yr["ambulances"].iloc[0]} ambulances</p>'
        '<p class="role">every 3-hour window replayed: plan with the forecast, score on what really happened.</p>'
        '<table class="yr"><thead><tr><th>placement</th><th>forecast</th><th>calls reached</th>'
        f'<th>urgent reached</th><th>avg km</th></tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )

st.markdown('<p class="footer">© 2026 Amanda Xuan</p>', unsafe_allow_html=True)