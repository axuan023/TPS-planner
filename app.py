"""
step 4b: the ambulance planner (version 1).

run: .venv/Scripts/python -m streamlit run app.py
needs: data/processed/forecast_2025.pkl, toronto_fsa.geojson, fsa_centroids.csv
style: style.css and .streamlit/config.toml (colours from amandaxuan.com)
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from planner import COVER_KM, coverage, distance_matrix, nearest_km, place_ambulances

# site colours
BG, TEXT, STRONG, FAINT, LINE, BOX, ACCENT = (
    "#131417", "#8b91a3", "#f2f3f5", "#4a4f5c", "#25282f", "#1c1e23", "#ef7a45")
FONT = "Roboto Mono, monospace"

st.set_page_config(page_title="TPS planner | Amanda Xuan", layout="wide")
st.markdown(f"<style>{Path('style.css').read_text()}</style>", unsafe_allow_html=True)

METHODS = {
    "machine learning (gradient boosting)": "gbm",
    "historical average": "hist_avg",
    "last 4 weeks": "last_4_weeks",
    "last week": "last_week",
}
SLOTS = ["12am–3am", "3am–6am", "6am–9am", "9am–12pm",
         "12pm–3pm", "3pm–6pm", "6pm–9pm", "9pm–12am"]


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
dist = distance_matrix(cent)
order = cent["fsa"].tolist()

# sidebar controls
st.sidebar.markdown('<p class="title">plan a shift</p>', unsafe_allow_html=True)
day = st.sidebar.date_input("date (2025)", value=dt.date(2025, 8, 1),
                            min_value=dt.date(2025, 1, 1),
                            max_value=dt.date(2025, 12, 31))
slot = st.sidebar.selectbox("time window", SLOTS, index=7)
n = st.sidebar.slider("ambulances available", 5, 60, 20)
method_label = st.sidebar.selectbox("forecast", list(METHODS))
method = METHODS[method_label]
show = st.sidebar.radio("map colour", ["forecast", "what actually happened"])

# pick the 3-hour block
block = pd.Timestamp(day) + pd.Timedelta(hours=3 * SLOTS.index(slot))
g = fc[fc["block"] == block].set_index("fsa").reindex(order).reset_index()

# header, same layout as the site
st.markdown(
    f'<p class="eyebrow">{block:%A, %B %d %Y}'.lower() + f' · {slot}</p>'
    '<p class="title">TPS planner</p>'
    '<p class="role">where should free ambulances wait, based on where calls are likely to come from.</p>',
    unsafe_allow_html=True,
)

if g["calls"].isna().all():
    st.markdown('<p class="note">no records for this window (system outage or daylight saving). '
                'pick another.</p>', unsafe_allow_html=True)
    st.stop()

# place ambulances using the forecast, and with hindsight for comparison
plan = place_ambulances(g[method].fillna(0), dist, n)
best = place_ambulances(g["calls"].fillna(0), dist, n)


def stat(label, value, accent=False):
    cls = "value accent" if accent else "value"
    return f'<div class="stat"><div class="label">{label}</div><div class="{cls}">{value}</div></div>'


st.markdown(
    '<div class="stats">'
    + stat("predicted calls", f"{g[method].sum():.0f}")
    + stat("actual calls", f"{g['calls'].sum():.0f}")
    + stat(f"calls within {COVER_KM:.0f} km", f"{coverage(g['calls'], dist, plan):.0%}", accent=True)
    + stat("hindsight plan", f"{coverage(g['calls'], dist, best):.0%}")
    + "</div>",
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
    'straight-line between area centres for now.</p>'
    '<p class="footer">© 2026 Amanda Xuan</p>',
    unsafe_allow_html=True,
)