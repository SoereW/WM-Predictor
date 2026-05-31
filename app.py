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
from src.sample_data import write_sample_data
from src.sample_squads import write_sample_squads
from src.squad import load_squads
from src.simulation import simulate_group

DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
MODEL_PATH = ROOT / "models" / "wm_predictor.joblib"

st.set_page_config(page_title="WM Predictor", page_icon="⚽", layout="wide")


def ensure_assets() -> None:
    if not DB_PATH.exists():
        data = ROOT / "data"
        real = data / "international_results.csv"
        sample = data / "sample_matches.csv"
        fixtures = data / "sample_fixtures_2026.csv"
        # Beispieldaten bei Bedarf erzeugen (nicht ins Repo eingecheckt).
        if not sample.exists() or not fixtures.exists():
            write_sample_data(data)

        con = connect(DB_PATH)
        init_schema(con)
        # Echte historische Daten bevorzugen, falls vorhanden; sonst Beispieldaten.
        load_matches(con, real if real.exists() else sample)
        load_fixtures(con, fixtures)
        con.close()
    if not MODEL_PATH.exists():
        con = connect(DB_PATH)
        matches = read_table(con, "matches")
        con.close()
        predictor = WMPredictor().fit(matches)
        # Bewertungssystem (Spieler + Chemie + Trainer) als Staerkequelle
        # anbinden; Fallback auf einfache Kaderstaerke.
        try:
            from src.rating.sample import COACHES, load_sample_players
            from src.rating.team import rate_all_teams

            predictor.attach_team_scores(rate_all_teams(load_sample_players(), coaches=COACHES))
        except Exception:
            squads_csv = ROOT / "data" / "sample_squads.csv"
            if not squads_csv.exists():
                write_sample_squads(ROOT / "data")
            predictor.attach_squads(load_squads(squads_csv))
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


@st.cache_data
def squads_for_tactics() -> pd.DataFrame:
    """Kaderdaten fuer das Taktik-Matchup (echte FIFA-Daten, sonst Demo)."""
    fifa = ROOT / "data" / "fifa_players.csv"
    if fifa.exists():
        from src.rating.fifa_ingest import load_fifa_as_squads

        return load_fifa_as_squads(fifa)
    from src.rating.sample import load_sample_players

    return load_sample_players()


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


ensure_assets()
matches, fixtures = load_data()
predictor = load_predictor()

st.title("⚽ WM Predictor Dashboard")
st.caption("Hybrid: Elo-Baseline + Kaderstärke + ML-Modell + Kontext-Adjustments. Beispiel-Daten enthalten; echte Datenquellen können angebunden werden.")

squad_active = bool(getattr(predictor, "squad_calib", None))
if squad_active:
    n_sq = len(getattr(predictor, "squad_overall", {}))
    pull = getattr(predictor, "squad_pull", 0.0)
    st.caption(
        f"🧮 Kaderstärke aktiv für {n_sq} Teams (Gewicht {pull:.0%}). "
        "Demo-Kaderdaten sind Näherungen – echte Kader-CSV via `--squads` einspeisbar."
    )

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

    st.header("Reise & Pause (Kontext)")
    home_rest = st.number_input(f"Ruhetage {fixture['home_team']}", 1, 14, 4)
    away_rest = st.number_input(f"Ruhetage {fixture['away_team']}", 1, 14, 4)
    away_travel_km = st.slider(f"Reisedistanz {fixture['away_team']} (km)", 0, 12000, 0, 500)
    venue_alt = st.slider("Stadionhöhe (m)", 0, 3600, 0, 100)

    auto_tactics = st.checkbox("Taktik-Matchup automatisch aus Kadern", value=True)

adjustments = {
    "player_delta": player_delta,
    "injury_delta": injury_delta,
    "rest_delta": rest_delta,
    "weather_delta": weather_delta,
}

# Kontext-Elo aus den Eingaben (Reise als Distanz des Gastes zum Spielort).
from src.context import altitude_delta, rest_delta as ctx_rest, travel_delta
import math as _math
ctx_elo = ctx_rest(home_rest, away_rest)
if away_travel_km > 0:
    # Naeherung: Gast reist away_travel_km, Heim 0 km zum Spielort.
    ctx_elo += travel_delta((0, 0), (0, away_travel_km / 111.0), (0, 0))
if venue_alt > 1200:
    ctx_elo += altitude_delta(venue_alt, 100.0, 1800.0)

# Taktik-Elo automatisch aus den Kadern (falls aktiviert und Daten vorhanden).
tactics_elo = 0.0
if auto_tactics:
    try:
        from src.rating.tactics import compute_tactics
        sq = squads_for_tactics()
        h = sq[sq["team"] == fixture["home_team"]]
        a = sq[sq["team"] == fixture["away_team"]]
        if len(h) and len(a):
            tactics_elo = compute_tactics(h, a).elo_delta
    except Exception:
        tactics_elo = 0.0

prediction = predictor.predict_fixture(
    fixture, matches, adjustments=adjustments,
    context_elo=ctx_elo, tactics_elo=tactics_elo,
)

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
st.plotly_chart(fig, width="stretch")

st.subheader("Wichtigste Modell-Features")
# Wert-Spalte als Text vereinheitlichen (mischt Zahlen und Labels wie "ja"/"nein").
factors_df = pd.DataFrame(prediction.top_factors, columns=["Feature", "Wert"])
factors_df["Wert"] = factors_df["Wert"].astype(str)
st.dataframe(factors_df, hide_index=True, width="stretch")

st.subheader("Gruppen-Simulation")
selected_group = fixture.get("group_name")
if selected_group:
    group_fixtures = fixtures[fixtures["group_name"] == selected_group]
    if len(group_fixtures) >= 2:
        sim = simulate_group(group_fixtures, matches, predictor, n=500)
        st.write(f"Top-2-Wahrscheinlichkeit auf Basis der verfügbaren Fixtures in Gruppe {selected_group}.")
        st.dataframe(sim, hide_index=True, width="stretch")
    else:
        st.info("Für diese Gruppe sind im Beispiel-Datensatz noch zu wenige Fixtures enthalten.")

st.subheader("Modelldiagnose")
st.json(predictor.training_summary)
st.warning(
    "Dieser Prototyp nutzt Beispiel-Daten. Für echte WM-Prognosen müssen vollständige historische Länderspiele, aktuelle Kader, erwartete Startelf, Verletzungen, Wetter und idealerweise Quoten ergänzt werden."
)
