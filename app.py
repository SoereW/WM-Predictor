"""WM-2026-Prognose-Dashboard.

Ein Befehl, alles laeuft: ``streamlit run app.py`` (oder ``python run.py``).
Zeigt Titelchancen, einen kompletten Turnierbaum, alle Gruppen- und K.-o.-
Vorhersagen, eine Einzelspiel-Analyse und eine optionale Eingabe fuer aktuelle
Ereignisse (Verletzungen/Sperren). Vorhersagen gewichten bewusst den
**aktuellen Kader** staerker als alte Laenderspielergebnisse.
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
        home, away, group = normalize_team(fx.home_team), normalize_team(fx.away_team), fx.group_name
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


# ===========================================================================
# Szenario berechnen (Kader-Gewicht + Verletzungen) -> alle Vorhersagen.
# ===========================================================================
@st.cache_data(show_spinner="Berechne Vorhersagen & Turnier ...")
def compute_scenario(squad_pull: float, injuries_text: str, n_sims: int) -> dict:
    """Erzeugt fuer ein Szenario: Gruppenspiel-Vorhersagen, Gruppentabellen-
    Wahrscheinlichkeiten, Titelchancen und den wahrscheinlichsten Turnierbaum.

    Greift fuer das Standardszenario (Default-Gewicht, keine Verletzungen) auf
    die vorgerechneten, eingecheckten CSVs zurueck (sofort), sonst wird live
    gerechnet.
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
    pred = copy.deepcopy(base_predictor())
    pred.squad_pull = float(squad_pull)
    squads = load_squads_df()
    if injuries_text.strip() and not squads.empty:
        edited, report = apply_injuries(squads, injuries_text)
        avail = edited[edited["available"].astype(str).str.lower().isin(["1", "true", "yes", "t", "ja"])]
        team_scores = rate_all_teams(avail)
        cov = coverage_confidence(squad_counts(avail))
        pred.attach_team_scores(team_scores, coverage=cov)

    prepared = ko.prepare_tournament(pred, history, GROUPS)
    preds = _predict_group_games(pred, fixtures, prepared, contexts)
    tour = ko.simulate_tournament(pred, history, GROUPS, fixtures=fixtures, n=n_sims, prepared=prepared)
    bracket, _ = ko.most_likely_bracket(pred, history, GROUPS, fixtures=fixtures, prepared=prepared)
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
    df = group_sim.rename(columns={"P(Platz 1)": "P(Platz 1)", "P(Top 2)": "P(Top 2)"})
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


# ===========================================================================
# Turnierbaum als Plotly-Figur.
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

    collect(104)
    y = {mid: float(i) for i, mid in enumerate(leaves)}

    def ypos(mid):
        if mid in ko.R32:
            return y[mid]
        (_, vh), (_, va) = ko.LATER[mid]
        yy = (ypos(vh) + ypos(va)) / 2.0
        y[mid] = yy
        return yy

    ypos(104)
    x = {mid: ko.ROUND_ORDER.index(ko.ROUND_OF[mid]) for mid in y}
    return x, y


def bracket_figure(bracket_df: pd.DataFrame) -> go.Figure:
    """Zeichnet den wahrscheinlichsten Turnierbaum (R32 -> Finale)."""
    x, y = _bracket_layout()
    by_id = {int(r["match_id"]): r for _, r in bracket_df.iterrows()}
    fig = go.Figure()

    # Verbindungslinien Kind -> Eltern.
    for mid, ((_, vh), (_, va)) in ko.LATER.items():
        for child in (vh, va):
            fig.add_trace(go.Scatter(
                x=[x[child], x[mid]], y=[-y[child], -y[mid]],
                mode="lines", line=dict(color="#3a4a63", width=1),
                hoverinfo="skip", showlegend=False,
            ))

    # Match-Kaesten (Heim/Gast, Sieger hervorgehoben).
    ann = []
    for mid, r in by_id.items():
        home, away, winner = r["home"], r["away"], r["winner"]
        p = float(r["p_home_advance"])
        ph = f"{p*100:.0f}%"
        pa = f"{(1-p)*100:.0f}%"
        hs = (f"<b>{home}</b>" if winner == home else f"<span style='color:#8895a7'>{home}</span>")
        as_ = (f"<b>{away}</b>" if winner == away else f"<span style='color:#8895a7'>{away}</span>")
        txt = f"{hs}  <i>{ph}</i><br>{as_}  <i>{pa}</i>"
        ann.append(dict(
            x=x[mid], y=-y[mid], text=txt, showarrow=False,
            xanchor="left", align="left", font=dict(size=10),
            bgcolor="#10192b", bordercolor="#2a3a55", borderwidth=1, borderpad=3,
        ))
    # Runden-Ueberschriften.
    for r in ko.ROUND_ORDER:
        xi = ko.ROUND_ORDER.index(r)
        ann.append(dict(x=xi, y=1.0, yref="paper", text=f"<b>{ko.ROUND_LABEL[r]}</b>",
                        showarrow=False, xanchor="left", font=dict(size=12, color="#cdd6e4")))
    # Weltmeister hervorheben.
    if 104 in by_id:
        champ = by_id[104]["winner"]
        ann.append(dict(x=len(ko.ROUND_ORDER) - 1, y=-y[104] - 1.1,
                        text=f"🏆 <b>{champ}</b>", showarrow=False, xanchor="left",
                        font=dict(size=15, color="#e0b341")))

    fig.update_layout(
        annotations=ann,
        xaxis=dict(visible=False, range=[-0.2, len(ko.ROUND_ORDER) + 0.3]),
        yaxis=dict(visible=False, range=[-16.2, 1.4]),
        height=1150, margin=dict(l=8, r=8, t=10, b=8),
        plot_bgcolor="#0b1220", paper_bgcolor="#0b1220",
    )
    return fig


def pct(v: float) -> str:
    return f"{float(v) * 100:.1f}%"


# ===========================================================================
# UI.
# ===========================================================================
ensure_assets()
predictor = base_predictor()
fixtures = load_fixtures_df()
team_ratings = load_csv("wm2026_team_ratings.csv")
player_ratings = load_csv("wm2026_player_ratings.csv")
tg = team_group()

st.title("🏆 WM 2026 – Vorhersage & Turnierbaum")
st.caption(
    "Hybrid aus aktueller **Kaderstärke** (EA-FC-26-Echtdaten, Stand 2025/26), "
    "World-Football-Elo (~49k echte Länderspiele) und einem kalibrierten Tor-/"
    "Klassifikationsmodell. Der **aktuelle Kader** zählt bewusst stärker als alte "
    "Ergebnisse – die hingen nur an den Spielern von damals."
)

# ---- Sidebar: Szenario --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Szenario")
    st.caption("Wie stark soll der **aktuelle Kader** zählen? Höher = aktuelle "
               "Spielerstärke dominiert, niedriger = mehr historische Form/Elo.")
    squad_pull = st.slider("Kaderstärke-Gewicht", 0.0, 0.8, DEFAULT_SQUAD_PULL, 0.05)

    st.subheader("🧑‍⚕️ Aktuelle Ereignisse")
    st.caption(
        "Optional. Eine Zeile pro Spieler im Format **`Team; Spielername`** "
        "(fehlt = nicht verfügbar) oder **`Team; Spielername; angeschlagen`** "
        "(Form −15 %). Beispiel siehe unten."
    )
    injuries_text = st.text_area(
        "Verletzungen / Sperren", value="", height=130,
        placeholder="Spain; Rodri\nFrance; Mbappe; angeschlagen\nEngland; Bellingham",
    )
    n_sims = st.select_slider("Simulationen (Turnier)", options=[1000, 2000, 4000, 8000], value=2000)
    apply = st.button("🔄 Szenario anwenden", width="stretch")

if apply:
    st.session_state["scenario"] = {"squad_pull": squad_pull, "injuries": injuries_text, "n": n_sims}

scen = st.session_state.get("scenario", {"squad_pull": DEFAULT_SQUAD_PULL, "injuries": "", "n": 2000})
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

tab_tour, tab_tree, tab_games, tab_groups, tab_match, tab_squads, tab_info = st.tabs([
    "🏆 Titelchancen", "🌳 Turnierbaum", "📋 Alle Spiele", "👥 Gruppen",
    "🔬 Einzelspiel", "🧑 Kader & Ratings", "ℹ️ Daten & Modell",
])

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
        marker_color="#e0b341",
    ))
    fig.update_layout(yaxis_tickformat=".0%", title="Titelwahrscheinlichkeit (Top 16)",
                      height=420, margin=dict(t=40, b=10))
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

# ---- Tab: Turnierbaum ---------------------------------------------------
with tab_tree:
    st.subheader("Wahrscheinlichster Turnierbaum")
    st.caption("Pro K.-o.-Spiel kommt der Favorit weiter (Prozent = Weiterkommens-"
               "Wahrscheinlichkeit). Die acht besten Gruppendritten werden gemäß "
               "FIFA-Bracket den Slots zugeordnet.")
    if not bracket_df.empty:
        st.plotly_chart(bracket_figure(bracket_df), width="stretch")
        with st.expander("Turnierbaum als Tabelle"):
            disp = bracket_df.rename(columns={
                "round_label": "Runde", "date": "Datum", "home": "Heim", "away": "Gast",
                "p_home_advance": "P(Heim weiter)", "winner": "Weiter",
            })[["Runde", "Datum", "Heim", "Gast", "P(Heim weiter)", "Weiter"]]
            st.dataframe(disp, hide_index=True, width="stretch")
    else:
        st.warning("Kein Turnierbaum verfügbar – bitte `python scripts/build_wm2026.py` ausführen.")

# ---- Tab: Alle Spiele ---------------------------------------------------
with tab_games:
    st.subheader("Alle 72 Gruppenspiele")
    pv = predictions.rename(columns={
        "date": "Datum", "group": "Gr.", "home_team": "Heim", "away_team": "Gast",
        "p_home": "1", "p_draw": "X", "p_away": "2", "xg_home": "xG H", "xg_away": "xG G", "tip": "Tipp",
    })
    cols = ["Datum", "Gr.", "Heim", "Gast", "1", "X", "2", "xG H", "xG G", "Tipp"]
    cols = [c for c in cols if c in pv.columns]
    st.dataframe(
        pv[cols], hide_index=True, width="stretch", height=460,
        column_config={c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.0f%%")
                       for c in ["1", "X", "2"] if c in pv.columns},
    )
    st.subheader("K.-o.-Spiele (wahrscheinlichster Baum)")
    if not bracket_df.empty:
        kv = bracket_df.rename(columns={
            "round_label": "Runde", "date": "Datum", "home": "Heim", "away": "Gast",
            "p_home_advance": "P(Heim weiter)", "winner": "Weiter",
        })
        st.dataframe(kv[["Runde", "Datum", "Heim", "Gast", "P(Heim weiter)", "Weiter"]],
                     hide_index=True, width="stretch")

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

# ---- Tab: Einzelspiel ---------------------------------------------------
with tab_match:
    st.subheader("Einzelspiel-Analyse mit Kontext-Reglern")
    history = load_history()
    flabel = fixtures.apply(lambda r: f"{r['date']} · {r['home_team']} vs {r['away_team']} · Gr. {r['group_name']}", axis=1)
    sel = st.selectbox("Spiel wählen", flabel.tolist())
    fx = fixtures.loc[flabel == sel].iloc[0].to_dict()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Manuelle Kontext-Anpassung** (positiv hilft Team 1):")
        player_delta = st.slider("Startelf/Form", -0.5, 0.5, 0.0, 0.05)
        injury_delta = st.slider("Verletzungen/Sperren", -0.5, 0.5, 0.0, 0.05)
    with c2:
        st.markdown("**Reise & Platz:**")
        home_rest = st.number_input(f"Ruhetage {fx['home_team']}", 1, 14, 4)
        away_rest = st.number_input(f"Ruhetage {fx['away_team']}", 1, 14, 4)
        away_travel = st.slider(f"Reisedistanz {fx['away_team']} (km)", 0, 12000, 0, 500)
        venue_alt = st.slider("Stadionhöhe (m)", 0, 3600, 0, 100)

    ctx = rest_delta(home_rest, away_rest)
    if away_travel > 0:
        ctx += travel_delta((0, 0), (0, away_travel / 111.0), (0, 0))
    if venue_alt > 1200:
        ctx += altitude_delta(venue_alt, 100.0, 1800.0)

    live = copy.deepcopy(predictor)
    live.squad_pull = scen["squad_pull"]
    pred = live.predict_fixture(
        fx, history, adjustments={"player_delta": player_delta, "injury_delta": injury_delta},
        context_elo=ctx,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{fx['home_team']} gewinnt", pct(pred.home_win))
    m2.metric("Unentschieden", pct(pred.draw))
    m3.metric(f"{fx['away_team']} gewinnt", pct(pred.away_win))
    g1, g2 = st.columns(2)
    g1.metric(f"xG {fx['home_team']}", f"{pred.expected_home_goals:.2f}")
    g2.metric(f"xG {fx['away_team']}", f"{pred.expected_away_goals:.2f}")

    fig = go.Figure(go.Bar(
        x=[fx["home_team"], "Remis", fx["away_team"]],
        y=[pred.home_win, pred.draw, pred.away_win],
        text=[pct(pred.home_win), pct(pred.draw), pct(pred.away_win)], textposition="auto",
        marker_color=["#4a90d9", "#9aa7b8", "#d96a4a"],
    ))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], height=320, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Modell-Faktoren"):
        ff = pd.DataFrame(pred.top_factors, columns=["Faktor", "Wert"])
        ff["Wert"] = ff["Wert"].astype(str)
        st.dataframe(ff, hide_index=True, width="stretch")

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
    real = (DATA / "international_results.csv").exists()
    st.markdown(
        f"""
- **Spieler/Kader:** EA FC 26 (Stand 2025/26), je Nation der stärkste verfügbare Kader
  → `data/wm2026_squads.csv` ({len(load_squads_df())} Spieler, 48 Nationen).
- **Historie:** {'martj42 (~49k echte Länderspiele)' if real else 'Beispieldaten'} – Training **strikt vor** dem
  WM-Start ({CUTOFF}), also leak-frei.
- **Auslosung & Spielplan:** offizielle Final-Auslosung (5.12.2025), 72 Gruppenspiele.
- **K.-o.-Baum:** offizielles FIFA-Bracket (Spiele 73–104), 8 beste Gruppendritte
  regelkonform zugeordnet.
- **Kadergewicht aktiv:** {scen['squad_pull']:.0%} (aktueller Kader korrigiert das historische Elo).
"""
    )
    st.subheader("Modelldiagnose")
    st.json(predictor.training_summary, expanded=False)
    st.caption("Reproduzieren: `python scripts/build_wm2026.py` (lädt echte Daten, baut alle Tabellen). "
               "Vergleich der Stärke-Signale: `python scripts/compare_systems.py`.")
