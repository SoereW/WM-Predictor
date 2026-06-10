"""WM-2026-Prognose-Dashboard – schlicht, modern, clean.

Ein Befehl, alles laeuft: ``python run.py`` (oder ``streamlit run app.py``).
Beim ersten Start baut der Bootstrap aus **echten Daten** Datenbank, Modell und
alle Prognose-Artefakte; danach zeigt das Dashboard:

* die Titelchancen jeder Nation,
* einen **kompletten Turnierbaum** (Sechzehntelfinale → Finale) mit den
  durchgerechneten Ergebnissen,
* eine **Detail-Prognose je Spiel** (auf ein Spiel im Baum klicken oder im
  Spiel-Center auswaehlen) und
* alle Gruppen-, K.-o.- und Kader-Tabellen.

Vorhersagen gewichten bewusst den **aktuellen Kader** staerker als alte
Laenderspielergebnisse.
"""

from __future__ import annotations

import copy
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import knockout as ko
from src.bootstrap import (
    CUTOFF, DATA, DB_PATH, DEFAULT_SQUAD_PULL, MODEL_PATH, SQUADS_CSV, ensure_assets,
)
from src.context import altitude_delta, rest_delta, travel_delta
from src.database import connect, read_table
from src.model import WMPredictor
from src.rating.squad_builder import coverage_confidence, squad_counts
from src.rating.team import rate_all_teams
from src.teams import normalize_team
from src.wm2026 import GROUPS, TEAM_COORD, fixture_venue_coord, team_group

st.set_page_config(page_title="WM 2026 Predictor", page_icon="🏆", layout="wide")

# --- Palette (schlicht, modern, clean) -------------------------------------
ACCENT = "#4f46e5"      # Indigo (Primaerfarbe)
WIN = "#059669"         # Smaragd (Sieger/positiv)
NEUTRAL = "#94a3b8"     # Schiefer (Remis/gedaempft)
GOLD = "#d97706"        # Bernstein (Weltmeister)
INK = "#1f2937"         # Text
LINE = "#e5e7eb"        # Rahmen/Linien
CARD = "#ffffff"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.1rem; max-width: 1300px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      [data-testid="stMetric"] {
          background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px;
          padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04);
      }
      [data-testid="stMetricLabel"] {color: #6b7280;}
      .stTabs [data-baseweb="tab-list"] {gap: 4px;}
      .stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 14px;}
      div[data-testid="stExpander"] {border-radius: 12px; border-color: #e5e7eb;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px;
             background:#eef2ff; color:#4f46e5; font-size:0.78rem; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Gecachte Loader.
# ===========================================================================
@st.cache_data
def load_history() -> pd.DataFrame:
    con = connect(DB_PATH)
    matches = read_table(con, "matches")
    con.close()
    train = matches[matches["date"] < CUTOFF]
    return train.reset_index(drop=True) if len(train) else matches.reset_index(drop=True)


@st.cache_data
def load_fixtures_df() -> pd.DataFrame:
    con = connect(DB_PATH)
    fx = read_table(con, "fixtures")
    con.close()
    return fx.sort_values(["date", "match_id"]).reset_index(drop=True)


@st.cache_data
def load_squads_df() -> pd.DataFrame:
    return pd.read_csv(SQUADS_CSV) if SQUADS_CSV.exists() else pd.DataFrame()


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    p = DATA / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_resource
def base_predictor() -> WMPredictor:
    return WMPredictor.load(MODEL_PATH)


@st.cache_data
def fixture_contexts() -> dict:
    """Kontext-Elo je Gruppenspiel (Reiseweg Heimat->Spielort + Ruhetage)."""
    from datetime import datetime

    fixtures = load_fixtures_df()
    last: dict = {}
    ctx: dict = {}
    for fx in fixtures.sort_values("date").itertuples(index=False):
        home, away = normalize_team(fx.home_team), normalize_team(fx.away_team)
        d = datetime.strptime(fx.date, "%Y-%m-%d")
        hr = (d - last[home]).days if home in last else None
        ar = (d - last[away]).days if away in last else None
        ce = rest_delta(hr, ar)
        ce += travel_delta(TEAM_COORD.get(home), TEAM_COORD.get(away), fixture_venue_coord(fx))
        ctx[int(fx.match_id)] = round(float(ce), 1)
        last[home] = d
        last[away] = d
    return ctx


# ===========================================================================
# Aktuelle Ereignisse: Verletzungen/Sperren aus Freitext anwenden.
# ===========================================================================
def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def parse_injuries(text: str):
    """Parst die Freitext-Eingabe in eine Liste (team, name, status)."""
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace("\t", ";").split(";")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        team = normalize_team(parts[0])
        name = parts[1]
        status = _strip_accents(parts[2]) if len(parts) > 2 and parts[2] else "out"
        doubtful = status in {"doubtful", "angeschlagen", "fraglich", "questionable"}
        entries.append((team, name, "doubtful" if doubtful else "out"))
    return entries


def apply_injuries(squads: pd.DataFrame, text: str):
    """Wendet Verletzungen an: ``out`` entfernt den Spieler, ``angeschlagen``
    senkt seine Form. Rueckgabe: (bearbeiteter Kader, Bericht-Liste)."""
    df = squads.copy()
    if "available" not in df.columns:
        df["available"] = 1
    name_norm = df["player"].map(_strip_accents)
    report = []
    for team, name, status in parse_injuries(text):
        q = _strip_accents(name)
        mask = (df["team"] == team) & name_norm.str.contains(q, regex=False)
        n = int(mask.sum())
        if n == 0:
            report.append((team, name, status, "nicht gefunden"))
            continue
        if status == "out":
            df.loc[mask, "available"] = 0
            report.append((team, name, "out", f"{n} Spieler entfernt"))
        else:
            df.loc[mask, "form"] = (pd.to_numeric(df.loc[mask, "form"], errors="coerce").fillna(70) * 0.85)
            report.append((team, name, "angeschlagen", f"{n} Spieler, Form -15%"))
    return df, report


@st.cache_resource(show_spinner=False)
def scenario_predictor(squad_pull: float, injuries_text: str) -> WMPredictor:
    """Predictor fuer ein Szenario (Kadergewicht + Verletzungen), gecacht.

    Wird sowohl fuer die Turnier-Berechnung als auch fuer jede Einzelspiel-
    Detailprognose genutzt, damit beides exakt zusammenpasst.
    """
    pred = copy.deepcopy(base_predictor())
    pred.squad_pull = float(squad_pull)
    squads = load_squads_df()
    if injuries_text.strip() and not squads.empty:
        edited, _ = apply_injuries(squads, injuries_text)
        avail = edited[edited["available"].astype(str).str.lower().isin(["1", "true", "yes", "t", "ja"])]
        pred.attach_team_scores(rate_all_teams(avail), coverage=coverage_confidence(squad_counts(avail)))
    return pred


# ===========================================================================
# Szenario berechnen (Kader-Gewicht + Verletzungen) -> alle Vorhersagen.
# ===========================================================================
@st.cache_data(show_spinner="Berechne Vorhersagen & Turnier ...")
def compute_scenario(squad_pull: float, injuries_text: str, n_sims: int) -> dict:
    """Erzeugt fuer ein Szenario: Gruppenspiel-Vorhersagen, Gruppentabellen-
    Wahrscheinlichkeiten, Titelchancen und den wahrscheinlichsten Turnierbaum.

    Standardszenario (Default-Gewicht, keine Verletzungen) -> vorgerechnete,
    eingecheckte CSVs (sofort); sonst wird live gerechnet.
    """
    history = load_history()
    fixtures = load_fixtures_df()
    contexts = fixture_contexts()
    report: list = []

    is_default = (abs(squad_pull - DEFAULT_SQUAD_PULL) < 1e-6) and not injuries_text.strip()
    pre_pred = load_csv("wm2026_predictions.csv")
    pre_ko = load_csv("wm2026_knockout_probs.csv")
    pre_bracket = load_csv("wm2026_bracket.csv")
    pre_group = load_csv("wm2026_group_sim.csv")
    if is_default and not pre_pred.empty and not pre_ko.empty and not pre_bracket.empty:
        return {
            "predictions": pre_pred, "knockout": pre_ko, "bracket": pre_bracket,
            "groups": _group_probs_from_sim(pre_group), "report": report,
            "champion": pre_bracket.loc[pre_bracket["round"] == "Final", "winner"].iloc[0]
            if (pre_bracket["round"] == "Final").any() else "",
            "source": "precomputed",
        }

    # --- Live: Predictor fuer dieses Szenario aufbauen ---------------------
    pred = scenario_predictor(squad_pull, injuries_text)
    if injuries_text.strip():
        _, report = apply_injuries(load_squads_df(), injuries_text)

    prepared = ko.prepare_tournament(pred, history, GROUPS)
    preds = _predict_group_games(pred, fixtures, prepared, contexts)
    tour = ko.simulate_tournament(pred, history, GROUPS, fixtures=fixtures, n=n_sims, prepared=prepared)
    bracket, _champ = ko.most_likely_bracket(pred, history, GROUPS, fixtures=fixtures, prepared=prepared)
    bracket_df = ko.bracket_to_frame(bracket)
    return {
        "predictions": preds, "knockout": tour.probs, "bracket": bracket_df,
        "groups": tour.groups, "report": report,
        "champion": bracket_df.loc[bracket_df["round"] == "Final", "winner"].iloc[0]
        if (bracket_df["round"] == "Final").any() else "",
        "source": "live",
    }


def _group_probs_from_sim(group_sim: pd.DataFrame) -> pd.DataFrame:
    """Bringt die vorgerechnete Gruppensimulation auf das einheitliche Schema."""
    if group_sim.empty:
        return group_sim
    df = group_sim.copy()
    df["P(weiter)"] = df["P(Top 2)"]
    if "P(Platz 2)" not in df.columns and {"P(Platz 1)", "P(Top 2)"} <= set(df.columns):
        df["P(Platz 2)"] = (df["P(Top 2)"] - df["P(Platz 1)"]).clip(lower=0).round(3)
    return df


def _predict_group_games(pred, fixtures, prepared, contexts) -> pd.DataFrame:
    """Alle Gruppenspiele aus den vorbereiteten Zustaenden vorhersagen (schnell)."""
    rows = []
    for _, fx in fixtures.iterrows():
        home, away = normalize_team(fx["home_team"]), normalize_team(fx["away_team"])
        ce = contexts.get(int(fx["match_id"]), 0.0)
        ph, pdr, pa, lh, la = pred.matchup_from_states(home, away, int(fx["neutral"]), prepared.states, extra_elo=ce)
        probs = {"1": ph, "X": pdr, "2": pa}
        rows.append({
            "date": fx["date"], "group": fx["group_name"], "home_team": home, "away_team": away,
            "p_home": round(ph, 3), "p_draw": round(pdr, 3), "p_away": round(pa, 3),
            "xg_home": round(lh, 2), "xg_away": round(la, 2),
            "tip": max(probs, key=probs.get), "context_elo": ce, "neutral": int(fx["neutral"]),
        })
    return pd.DataFrame(rows)


def pct(v: float) -> str:
    return f"{float(v) * 100:.1f}%"


# ===========================================================================
# Turnierbaum als Plotly-Figur (clean, hover-/klickbar).
# ===========================================================================
def _bracket_layout():
    """Baum-Positionen je Spiel-ID: x = Runde, y = vertikale Position."""
    leaves = []

    def collect(mid):
        if mid in ko.R32:
            leaves.append(mid)
            return
        (_, vh), (_, va) = ko.LATER[mid]
        collect(vh)
        collect(va)

    collect(ko.FINAL)
    y = {mid: float(i) for i, mid in enumerate(leaves)}

    def ypos(mid):
        if mid in ko.R32:
            return y[mid]
        (_, vh), (_, va) = ko.LATER[mid]
        yy = (ypos(vh) + ypos(va)) / 2.0
        y[mid] = yy
        return yy

    ypos(ko.FINAL)
    x = {mid: ko.ROUND_ORDER.index(ko.ROUND_OF[mid]) for mid in y}
    return x, y


def bracket_figure(bracket_df: pd.DataFrame) -> go.Figure:
    """Zeichnet den durchgerechneten Turnierbaum (R32 -> Finale), clean & klickbar."""
    x, y = _bracket_layout()
    by_id = {int(r["match_id"]): r for _, r in bracket_df.iterrows()}
    fig = go.Figure()
    cw, ch = 0.92, 0.40  # Kartenbreite/-hoehe (in Datenkoordinaten)

    # Verbindungslinien (Kind-Rechts -> Eltern-Links).
    for mid, ((_, vh), (_, va)) in ko.LATER.items():
        for child in (vh, va):
            fig.add_trace(go.Scatter(
                x=[x[child] + cw, x[mid]], y=[-y[child], -y[mid]],
                mode="lines", line=dict(color=LINE, width=1.4, shape="spline"),
                hoverinfo="skip", showlegend=False,
            ))

    node_x, node_y, node_cd, node_hover = [], [], [], []
    for mid, r in by_id.items():
        home, away, winner = r["home"], r["away"], r["winner"]
        p = float(r["p_home_advance"])
        ph, pa = f"{p*100:.0f}%", f"{(1-p)*100:.0f}%"
        x0, x1 = x[mid] + 0.02, x[mid] + cw
        y0, y1 = -y[mid] - ch, -y[mid] + ch
        is_final = mid == ko.FINAL
        # Karte.
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=GOLD if is_final else LINE, width=1.6 if is_final else 1),
                      fillcolor="#fffdf5" if is_final else CARD, layer="below")
        # Sieger oben hervorgehoben, Verlierer gedaempft.
        h_color, h_w = (WIN, "bold") if winner == home else (NEUTRAL, "normal")
        a_color, a_w = (WIN, "bold") if winner == away else (NEUTRAL, "normal")
        txt = (f"<span style='color:{h_color};font-weight:{h_w}'>{home}</span>"
               f"<span style='color:#9ca3af'> {ph}</span><br>"
               f"<span style='color:{a_color};font-weight:{a_w}'>{away}</span>"
               f"<span style='color:#9ca3af'> {pa}</span>")
        fig.add_annotation(x=x0 + 0.03, y=-y[mid], text=txt, showarrow=False,
                           xanchor="left", align="left", font=dict(size=11, color=INK),
                           captureevents=False)
        # Transparenter Klick-/Hover-Punkt in der Kartenmitte.
        node_x.append((x0 + x1) / 2.0)
        node_y.append(-y[mid])
        node_cd.append([mid])
        node_hover.append(f"{ko.ROUND_LABEL[r['round']]}<br><b>{home}</b> {ph}  ·  <b>{away}</b> {pa}"
                          f"<br>→ weiter: <b>{winner}</b>")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers", marker=dict(size=30, symbol="square", color="rgba(0,0,0,0)"),
        customdata=node_cd, hovertext=node_hover, hoverinfo="text", showlegend=False,
    ))

    # Runden-Ueberschriften.
    for r in ko.ROUND_ORDER:
        xi = ko.ROUND_ORDER.index(r)
        fig.add_annotation(x=xi + 0.45, y=1.012, yref="paper", text=ko.ROUND_LABEL[r],
                           showarrow=False, xanchor="center",
                           font=dict(size=12, color=ACCENT, family="sans serif"))
    # Weltmeister.
    if ko.FINAL in by_id:
        fig.add_annotation(x=len(ko.ROUND_ORDER) - 1 + 0.45, y=-y[ko.FINAL] - 1.25,
                           text=f"🏆 {by_id[ko.FINAL]['winner']}", showarrow=False, xanchor="center",
                           font=dict(size=17, color=GOLD))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.1, len(ko.ROUND_ORDER) + 0.05], fixedrange=True),
        yaxis=dict(visible=False, range=[-16.4, 1.2], fixedrange=True),
        height=1120, margin=dict(l=6, r=6, t=18, b=6),
        plot_bgcolor=CARD, paper_bgcolor=CARD, hovermode="closest",
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font_size=12),
    )
    return fig


# ===========================================================================
# Detail-Prognose fuer ein einzelnes Spiel.
# ===========================================================================
def _top_scorelines(grid: np.ndarray, k: int = 4):
    flat = [((h, a), float(grid[h, a])) for h in range(grid.shape[0]) for a in range(grid.shape[1])]
    flat.sort(key=lambda t: t[1], reverse=True)
    return flat[:k]


def render_match_detail(pred, history, home, away, neutral, *, is_ko=False,
                        context_elo=0.0, adjustments=None, header=None):
    """Reiche Detail-Prognose fuer ein Duell (Gruppe oder K.-o.)."""
    home, away = normalize_team(home), normalize_team(away)
    fixture = {"home_team": home, "away_team": away, "neutral": int(neutral)}
    adjustments = adjustments or {}
    p = pred.predict_fixture(fixture, history, adjustments=adjustments, context_elo=context_elo)

    if header:
        st.markdown(f"#### {header}")
    st.markdown(f"### {home} &nbsp;–&nbsp; {away}")

    if is_ko:
        adv = ko._advance_prob(p.home_win, p.draw, p.away_win)
        c1, c2 = st.columns(2)
        c1.metric(f"➡️ {home} kommt weiter", pct(adv))
        c2.metric(f"➡️ {away} kommt weiter", pct(1 - adv))

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Sieg {home}", pct(p.home_win))
    m2.metric("Unentschieden", pct(p.draw))
    m3.metric(f"Sieg {away}", pct(p.away_win))

    fig = go.Figure(go.Bar(
        x=[p.home_win, p.draw, p.away_win], y=[home, "Remis", away], orientation="h",
        text=[pct(p.home_win), pct(p.draw), pct(p.away_win)], textposition="auto",
        marker_color=[ACCENT, NEUTRAL, "#fb7185"],
    ))
    fig.update_layout(xaxis_tickformat=".0%", xaxis_range=[0, 1], height=190,
                      margin=dict(t=6, b=6, l=6, r=6), plot_bgcolor=CARD, paper_bgcolor=CARD,
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    g1, g2 = st.columns([1, 1.3])
    with g1:
        st.metric(f"Erwartete Tore {home}", f"{p.expected_home_goals:.2f}")
        st.metric(f"Erwartete Tore {away}", f"{p.expected_away_goals:.2f}")
    with g2:
        grid = pred.score_matrix(fixture, history, adjustments=adjustments)
        st.caption("**Wahrscheinlichste Ergebnisse**")
        lines = [f"<span class='pill'>{h}:{a}</span> &nbsp;{pr*100:.1f}%"
                 for (h, a), pr in _top_scorelines(grid, 4)]
        st.markdown("<br>".join(lines), unsafe_allow_html=True)

    with st.expander("Modell-Faktoren"):
        ff = pd.DataFrame(p.top_factors, columns=["Faktor", "Wert"])
        ff["Wert"] = ff["Wert"].astype(str)
        st.dataframe(ff, hide_index=True, width="stretch")


# ===========================================================================
# Initialisierung + gemeinsame Daten.
# ===========================================================================
ensure_assets()
predictor = base_predictor()
fixtures = load_fixtures_df()
team_ratings = load_csv("wm2026_team_ratings.csv")
player_ratings = load_csv("wm2026_player_ratings.csv")
tg = team_group()

st.title("🏆 WM 2026 – Vorhersage & Turnierbaum")
st.caption(
    "Hybrid aus aktueller **Kaderstärke** (EA-FC-26-Echtdaten), World-Football-Elo "
    "(~49 000 echte Länderspiele) und einem kalibrierten Tor-/Klassifikationsmodell. "
    "Der **aktuelle Kader** zählt bewusst stärker als alte Ergebnisse."
)

# ---- Sidebar: Szenario --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Szenario")
    st.caption("Wie stark zählt der **aktuelle Kader**? Höher = aktuelle "
               "Spielerstärke dominiert, niedriger = mehr Historie/Elo.")
    squad_pull = st.slider("Kaderstärke-Gewicht", 0.0, 0.8, DEFAULT_SQUAD_PULL, 0.05)

    st.subheader("🧑‍⚕️ Aktuelle Ereignisse")
    st.caption("Optional. Eine Zeile je Spieler: **`Team; Spielername`** "
               "(fehlt) oder **`Team; Spielername; angeschlagen`** (Form −15 %).")
    injuries_text = st.text_area(
        "Verletzungen / Sperren", value="", height=120,
        placeholder="Spain; Rodri\nFrance; Mbappe; angeschlagen\nEngland; Bellingham",
    )
    n_sims = st.select_slider("Simulationen (Turnier)", options=[1000, 2000, 4000, 8000], value=4000)
    apply = st.button("🔄 Szenario anwenden", width="stretch")

if apply:
    st.session_state["scenario"] = {"squad_pull": squad_pull, "injuries": injuries_text, "n": n_sims}

scen = st.session_state.get("scenario", {"squad_pull": DEFAULT_SQUAD_PULL, "injuries": "", "n": 4000})
result = compute_scenario(scen["squad_pull"], scen["injuries"], scen["n"])

if result["source"] == "live":
    st.info(f"🔬 **Szenario aktiv** – Kadergewicht {scen['squad_pull']:.0%}"
            + (", inkl. eingetragener Ereignisse" if scen["injuries"].strip() else "")
            + f" · {scen['n']} Simulationen.")
    if result["report"]:
        with st.expander("Angewandte Ereignisse", expanded=True):
            rep = pd.DataFrame(result["report"], columns=["Team", "Eingabe", "Status", "Wirkung"])
            st.dataframe(rep, hide_index=True, width="stretch")

predictions = result["predictions"]
knockout = result["knockout"]
bracket_df = result["bracket"]
group_probs = result["groups"]
champion = result["champion"]

# Predictor + Historie fuer Live-Detailprognosen (passend zum Szenario).
detail_pred = scenario_predictor(scen["squad_pull"], scen["injuries"])
history_df = load_history()

tab_tree, tab_tour, tab_games, tab_groups, tab_match, tab_squads, tab_info = st.tabs([
    "🌳 Turnierbaum", "🏆 Titelchancen", "📋 Alle Spiele", "👥 Gruppen",
    "🔬 Spiel-Center", "🧑 Kader & Ratings", "ℹ️ Daten & Modell",
])

# ---- Tab: Turnierbaum (Centerpiece, klickbar) ---------------------------
with tab_tree:
    cL, cR = st.columns([3, 0.9])
    with cL:
        st.subheader("Durchgerechneter Turnierbaum")
        st.caption("Pro K.-o.-Spiel kommt der Favorit weiter (Prozent = Weiterkommens-"
                   "Chance). **Auf ein Spiel zeigen** für die Kurzinfo, **klicken** für die "
                   "Detail-Prognose darunter.")
    with cR:
        if champion:
            cp = knockout.loc[knockout["team"] == champion, "P(Titel)"]
            st.metric("🏆 Weltmeister-Prognose", champion,
                      help="Wahrscheinlichster Turnierbaum")
            if len(cp):
                st.caption(f"Titelchance: **{pct(cp.iloc[0])}**")

    if not bracket_df.empty:
        event = st.plotly_chart(bracket_figure(bracket_df), width="stretch",
                                key="bracket_tree", on_select="rerun")
        # Klick im Baum auswerten.
        clicked = None
        try:
            pts = event["selection"]["points"] if isinstance(event, dict) else event.selection["points"]
            if pts:
                clicked = int(pts[-1]["customdata"][0])
        except Exception:  # noqa: BLE001
            clicked = None
        if clicked is not None:
            st.session_state["tree_match"] = clicked

        ids = sorted(bracket_df["match_id"].astype(int))
        labels = {int(r["match_id"]): f"{r['round_label']}: {r['home']} – {r['away']}"
                  for _, r in bracket_df.iterrows()}
        default_mid = st.session_state.get("tree_match", ids[-1])
        if default_mid not in ids:
            default_mid = ids[-1]
        st.divider()
        sel_mid = st.selectbox(
            "Spiel für Detail-Prognose", ids, index=ids.index(default_mid),
            format_func=lambda m: labels.get(int(m), str(m)),
            key="tree_select",
        )
        st.session_state["tree_match"] = sel_mid
        row = bracket_df.loc[bracket_df["match_id"].astype(int) == int(sel_mid)].iloc[0]
        with st.container(border=True):
            render_match_detail(detail_pred, history_df, row["home"], row["away"], neutral=1,
                                is_ko=True, header=f"{row['round_label']} · {row['date']}")
    else:
        st.warning("Kein Turnierbaum verfügbar – bitte `python scripts/build_wm2026.py` ausführen.")

# ---- Tab: Titelchancen --------------------------------------------------
with tab_tour:
    if champion:
        cp = knockout.loc[knockout["team"] == champion, "P(Titel)"]
        st.subheader(f"Prognostizierter Weltmeister: **{champion}** "
                     + (f"· {pct(cp.iloc[0])} Titelchance" if len(cp) else ""))
    top = knockout.head(8)
    cols = st.columns(4)
    for i, (_, r) in enumerate(top.head(4).iterrows()):
        cols[i].metric(f"{i+1}. {r['team']}", pct(r["P(Titel)"]), help="Titelwahrscheinlichkeit")
    cols = st.columns(4)
    for i, (_, r) in enumerate(top.tail(4).iterrows()):
        cols[i].metric(f"{i+5}. {r['team']}", pct(r["P(Titel)"]))

    fig = go.Figure(go.Bar(
        x=knockout.head(16)["team"], y=knockout.head(16)["P(Titel)"],
        text=[pct(v) for v in knockout.head(16)["P(Titel)"]], textposition="outside",
        marker_color=ACCENT,
    ))
    fig.update_layout(yaxis_tickformat=".0%", title="Titelwahrscheinlichkeit (Top 16)",
                      height=420, margin=dict(t=40, b=10), plot_bgcolor=CARD, paper_bgcolor=CARD)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Wie weit kommt jede Nation? (Wahrscheinlichkeit je Runde)**")
    st.dataframe(
        knockout.rename(columns={
            "rank": "Rang", "team": "Team", "group": "Gr.", "P(R32)": "Sechzehntelf.",
            "P(R16)": "Achtelf.", "P(QF)": "Viertelf.", "P(SF)": "Halbf.",
            "P(Finale)": "Finale", "P(Titel)": "Titel",
        }),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.0f%%")
                       for c in ["Sechzehntelf.", "Achtelf.", "Viertelf.", "Halbf.", "Finale", "Titel"]},
    )

# ---- Tab: Alle Spiele ---------------------------------------------------
with tab_games:
    st.subheader("Alle 72 Gruppenspiele")
    pv = predictions.rename(columns={
        "date": "Datum", "group": "Gr.", "home_team": "Heim", "away_team": "Gast",
        "p_home": "1", "p_draw": "X", "p_away": "2", "xg_home": "xG H", "xg_away": "xG G", "tip": "Tipp",
    })
    cols = [c for c in ["Datum", "Gr.", "Heim", "Gast", "1", "X", "2", "xG H", "xG G", "Tipp"] if c in pv.columns]
    st.dataframe(
        pv[cols], hide_index=True, width="stretch", height=460,
        column_config={c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.0f%%")
                       for c in ["1", "X", "2"] if c in pv.columns},
    )
    st.subheader("K.-o.-Spiele (durchgerechneter Baum)")
    if not bracket_df.empty:
        kv = bracket_df.rename(columns={
            "round_label": "Runde", "date": "Datum", "home": "Heim", "away": "Gast",
            "p_home_advance": "P(Heim weiter)", "winner": "Weiter",
        })
        st.dataframe(kv[["Runde", "Datum", "Heim", "Gast", "P(Heim weiter)", "Weiter"]],
                     hide_index=True, width="stretch",
                     column_config={"P(Heim weiter)": st.column_config.ProgressColumn(
                         "P(Heim weiter)", min_value=0.0, max_value=1.0, format="%.0f%%")})

# ---- Tab: Gruppen -------------------------------------------------------
with tab_groups:
    st.subheader("Gruppen – Weiterkommens-Wahrscheinlichkeiten")
    st.caption("Pro Gruppe die Wahrscheinlichkeit, als 1./2. weiterzukommen (Monte Carlo).")
    gp = group_probs
    group_list = sorted(GROUPS.keys())
    for i in range(0, len(group_list), 3):
        row = group_list[i:i + 3]
        cc = st.columns(len(row))
        for k, g in enumerate(row):
            with cc[k]:
                st.markdown(f"**Gruppe {g}**")
                sub = gp[gp["group"] == g] if (not gp.empty and "group" in gp.columns) else pd.DataFrame()
                if sub.empty:
                    st.caption("– keine Daten –")
                    continue
                want = [c for c in ["team", "Team", "P(Platz 1)", "P(Top 2)"] if c in sub.columns]
                sub = sub[want].rename(columns={"team": "Team", "P(Platz 1)": "P1", "P(Top 2)": "Top2"})
                if "Top2" in sub.columns:
                    sub = sub.sort_values("Top2", ascending=False)
                st.dataframe(
                    sub, hide_index=True, width="stretch",
                    column_config={c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.0f%%")
                                   for c in ["P1", "Top2"] if c in sub.columns},
                )

# ---- Tab: Spiel-Center (jedes Spiel + Kontext) --------------------------
with tab_match:
    st.subheader("Spiel-Center – Detailprognose für jedes Spiel")
    st.caption("Wähle ein Gruppen- **oder** K.-o.-Spiel und passe den Kontext an.")

    kind = st.radio("Spieltyp", ["Gruppenspiel", "K.-o.-Spiel"], horizontal=True)
    if kind == "Gruppenspiel":
        flabel = fixtures.apply(
            lambda r: f"{r['date']} · {r['home_team']} – {r['away_team']} · Gr. {r['group_name']}", axis=1)
        sel = st.selectbox("Spiel wählen", flabel.tolist())
        fx = fixtures.loc[flabel == sel].iloc[0].to_dict()
        sel_home, sel_away, sel_neutral, sel_ko = fx["home_team"], fx["away_team"], int(fx["neutral"]), False
        sel_header = f"Gruppe {fx['group_name']} · {fx['date']}"
    else:
        if bracket_df.empty:
            st.info("Kein Turnierbaum verfügbar.")
            st.stop()
        klabel = bracket_df.apply(lambda r: f"{r['round_label']} · {r['home']} – {r['away']}", axis=1)
        sel = st.selectbox("K.-o.-Spiel wählen", klabel.tolist())
        kr = bracket_df.loc[klabel == sel].iloc[0]
        sel_home, sel_away, sel_neutral, sel_ko = kr["home"], kr["away"], 1, True
        sel_header = f"{kr['round_label']} · {kr['date']}"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Manuelle Anpassung** (positiv hilft Team 1):")
        player_delta = st.slider("Startelf/Form", -0.5, 0.5, 0.0, 0.05)
        injury_delta = st.slider("Verletzungen/Sperren", -0.5, 0.5, 0.0, 0.05)
    with c2:
        st.markdown("**Reise & Platz:**")
        home_rest = st.number_input(f"Ruhetage {sel_home}", 1, 14, 4)
        away_rest = st.number_input(f"Ruhetage {sel_away}", 1, 14, 4)
        away_travel = st.slider(f"Reisedistanz {sel_away} (km)", 0, 12000, 0, 500)
        venue_alt = st.slider("Stadionhöhe (m)", 0, 3600, 0, 100)

    ctx = rest_delta(home_rest, away_rest)
    if away_travel > 0:
        ctx += travel_delta((0, 0), (0, away_travel / 111.0), (0, 0))
    if venue_alt > 1200:
        ctx += altitude_delta(venue_alt, 100.0, 1800.0)

    with st.container(border=True):
        render_match_detail(
            detail_pred, history_df, sel_home, sel_away, neutral=sel_neutral, is_ko=sel_ko,
            context_elo=ctx, adjustments={"player_delta": player_delta, "injury_delta": injury_delta},
            header=sel_header,
        )

# ---- Tab: Kader & Ratings ----------------------------------------------
with tab_squads:
    st.subheader("Team-Rangliste (aktuelle Kaderstärke)")
    if not team_ratings.empty:
        tr = team_ratings.rename(columns={
            "rank": "Rang", "group": "Gr.", "team": "Team", "overall": "Gesamt",
            "best_xi": "Beste XI", "depth": "Tiefe", "chemistry": "Chemie",
            "coach": "Trainer", "n_players": "Spieler",
        })
        st.dataframe(tr, hide_index=True, width="stretch", height=420)
    st.subheader("Spieler einer Nation")
    if not player_ratings.empty:
        nation = st.selectbox("Nation", sorted(player_ratings["team"].unique()))
        sub = player_ratings[player_ratings["team"] == nation].sort_values("overall", ascending=False)
        sub = sub.rename(columns={"player": "Spieler", "position": "Pos.", "club": "Verein",
                                  "age": "Alter", "overall": "Gesamt", "skill": "Skill", "form": "Form"})
        keep = [c for c in ["Spieler", "Pos.", "Verein", "Alter", "Gesamt", "Skill", "Form"] if c in sub.columns]
        st.dataframe(sub[keep], hide_index=True, width="stretch", height=420)

# ---- Tab: Daten & Modell ------------------------------------------------
with tab_info:
    st.subheader("Datengrundlage")
    st.markdown(
        f"""
- **Spieler/Kader:** EA FC 26 (Stand 2025/26), je Nation der stärkste verfügbare Kader
  → `data/wm2026_squads.csv` ({len(load_squads_df())} Spieler, 48 Nationen).
- **Historie:** martj42 (~49 000 echte Länderspiele), automatisch beim ersten Start
  geladen – Training **strikt vor** dem WM-Start ({CUTOFF}), also leak-frei.
- **Auslosung & Spielplan:** Final-Auslosung (5.12.2025), 72 Gruppenspiele.
- **K.-o.-Baum:** FIFA-Format (Spiele 73–104), 8 beste Gruppendritte regelkonform
  zugeordnet, durchgerechnet bis zum Finale.
- **Kadergewicht aktiv:** {scen['squad_pull']:.0%} (aktueller Kader korrigiert das historische Elo).
"""
    )
    st.subheader("Modelldiagnose")
    st.json(predictor.training_summary, expanded=False)
    st.caption("Reproduzieren: `python scripts/build_wm2026.py` (lädt echte Daten, baut alle Tabellen). "
               "Erststart/Init: `python run.py`.")
