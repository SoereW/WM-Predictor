from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.database import connect, init_schema, load_fixtures, load_matches, read_table
from src.model import WMPredictor
from src.simulation import simulate_group

DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
MODEL_PATH = ROOT / "models" / "wm_predictor.joblib"

st.set_page_config(page_title="WM Predictor", page_icon="⚽", layout="wide")


def ensure_assets() -> None:
    if not DB_PATH.exists():
        con = connect(DB_PATH)
        init_schema(con)
        load_matches(con, ROOT / "data" / "sample_matches.csv")
        load_fixtures(con, ROOT / "data" / "sample_fixtures_2026.csv")
        con.close()
    if not MODEL_PATH.exists():
        con = connect(DB_PATH)
        matches = read_table(con, "matches")
        con.close()
        predictor = WMPredictor().fit(matches)
        predictor.save(MODEL_PATH)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = connect(DB_PATH)
    matches = read_table(con, "matches")
    fixtures = read_table(con, "fixtures")
    con.close()
    return matches, fixtures


@st.cache_resource
def load_predictor() -> WMPredictor:
    return WMPredictor.load(MODEL_PATH)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


ensure_assets()
matches, fixtures = load_data()
predictor = load_predictor()

st.title("⚽ WM Predictor Dashboard")
st.caption("Hybrid-Prototyp: Elo-Baseline + ML-Modell + Kontext-Adjustments. Beispiel-Daten enthalten; echte Datenquellen können angebunden werden.")

with st.sidebar:
    st.header("Spiel auswählen")
    fixtures = fixtures.sort_values(["date", "match_id"])
    fixture_label = fixtures.apply(
        lambda r: f"{r['date']} · {r['home_team']} vs {r['away_team']} · {r['city']}", axis=1
    )
    selected_label = st.selectbox("Fixture", fixture_label.tolist())
    fixture = fixtures.loc[fixture_label == selected_label].iloc[0].to_dict()

    st.header("Kontext-Adjustments")
    st.write("Positive Werte helfen Team 1, negative Werte Team 2.")
    player_delta = st.slider("Startelf-/Spielerform-Vorteil", -0.50, 0.50, 0.00, 0.05)
    injury_delta = st.slider("Verletzungen/Sperren", -0.50, 0.50, 0.00, 0.05)
    rest_delta = st.slider("Erholung/Reise", -0.30, 0.30, 0.00, 0.05)
    weather_delta = st.slider("Wetter/Stil-Matchup", -0.25, 0.25, 0.00, 0.05)

adjustments = {
    "player_delta": player_delta,
    "injury_delta": injury_delta,
    "rest_delta": rest_delta,
    "weather_delta": weather_delta,
}
prediction = predictor.predict_fixture(fixture, matches, adjustments=adjustments)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"{fixture['home_team']} gewinnt", pct(prediction.home_win))
with col2:
    st.metric("Unentschieden", pct(prediction.draw))
with col3:
    st.metric(f"{fixture['away_team']} gewinnt", pct(prediction.away_win))

st.subheader("Erwartete Tore")
goals_col1, goals_col2 = st.columns(2)
with goals_col1:
    st.metric(f"xG {fixture['home_team']}", f"{prediction.expected_home_goals:.2f}")
with goals_col2:
    st.metric(f"xG {fixture['away_team']}", f"{prediction.expected_away_goals:.2f}")

fig = go.Figure(
    data=[
        go.Bar(
            x=[fixture["home_team"], "Unentschieden", fixture["away_team"]],
            y=[prediction.home_win, prediction.draw, prediction.away_win],
            text=[pct(prediction.home_win), pct(prediction.draw), pct(prediction.away_win)],
            textposition="auto",
        )
    ]
)
fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], title="1X2-Wahrscheinlichkeiten")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Wichtigste Modell-Features")
st.dataframe(
    pd.DataFrame(prediction.top_factors, columns=["Feature", "Wert"]),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Gruppen-Simulation")
selected_group = fixture.get("group_name")
if selected_group:
    group_fixtures = fixtures[fixtures["group_name"] == selected_group]
    if len(group_fixtures) >= 2:
        sim = simulate_group(group_fixtures, matches, predictor, n=500)
        st.write(f"Top-2-Wahrscheinlichkeit auf Basis der verfügbaren Fixtures in Gruppe {selected_group}.")
        st.dataframe(sim, hide_index=True, use_container_width=True)
    else:
        st.info("Für diese Gruppe sind im Beispiel-Datensatz noch zu wenige Fixtures enthalten.")

st.subheader("Modelldiagnose")
st.json(predictor.training_summary)
st.warning(
    "Dieser Prototyp nutzt Beispiel-Daten. Für echte WM-Prognosen müssen vollständige historische Länderspiele, aktuelle Kader, erwartete Startelf, Verletzungen, Wetter und idealerweise Quoten ergänzt werden."
)
