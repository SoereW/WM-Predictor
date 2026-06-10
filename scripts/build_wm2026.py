"""End-to-End-Pipeline mit ECHTEN Daten fuer die WM 2026.

Ablauf (alle Schritte mit echten, online geladenen Daten):

1. Echte Spielerdaten laden (EA FC 26, Stand 2025/26) und je WM-Nation den
   staerksten verfuegbaren Kader zusammenstellen.
2. Jeden Spieler bewerten -> Spieler-Gesamtrating; je Team aggregieren
   (beste Elf + Tiefe + Chemie + Trainer) -> Team-Gesamtrating.
3. Echte Historie (martj42, ~49k Laenderspiele) als Elo-/Form-Basis laden,
   ein Modell **vor WM-Beginn** trainieren (leak-frei) und die Team-Ratings
   als Kaderstaerke einkoppeln.
4. Alle 72 Gruppenspiele der echten Auslosung vorhersagen (1X2 + xG) und je
   Gruppe die Weiterkommens-Wahrscheinlichkeiten simulieren.

Ergebnis-Artefakte landen in ``data/`` (CSV) und werden zusammengefasst
ausgegeben. Modell + DB werden gespeichert, sodass das Dashboard dieselben
echten Daten nutzt.

    python scripts/build_wm2026.py            # nutzt vorhandene/online Daten
    python scripts/build_wm2026.py --refresh  # Quelldaten neu herunterladen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context import rest_delta, travel_delta
from src.database import connect, init_schema, load_fixtures, load_matches, read_table
from src.ingest import fetch_international_results
from src.model import WMPredictor
from src.rating.fifa_ingest import fetch_fifa_players, load_fifa_as_squads
from src.rating.player import rate_player
from src.rating.squad_builder import build_squads, coverage_confidence, squad_counts
from src.rating.team import rate_all_teams
from src.simulation import simulate_group
from src.wm2026 import (
    GROUPS,
    GROUP_STAGE_START,
    TEAM_COORD,
    all_teams,
    team_group,
    venue_coord,
    write_fixtures,
)

DATA = ROOT / "data"
DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
MODEL_PATH = ROOT / "models" / "wm_predictor.joblib"

FC26_CSV = DATA / "fc26_players.csv"
HISTORY_CSV = DATA / "international_results.csv"

SQUADS_OUT = DATA / "wm2026_squads.csv"
PLAYERS_OUT = DATA / "wm2026_player_ratings.csv"
TEAMS_OUT = DATA / "wm2026_team_ratings.csv"
FIXTURES_OUT = DATA / "wm2026_fixtures.csv"
PRED_OUT = DATA / "wm2026_predictions.csv"
SIM_OUT = DATA / "wm2026_group_sim.csv"
KNOCKOUT_OUT = DATA / "wm2026_knockout_probs.csv"
BRACKET_OUT = DATA / "wm2026_bracket.csv"

# Standardgewicht der Kaderstaerke (deckungsgleich mit dem Dashboard-Default).
DEFAULT_SQUAD_PULL = 0.7

# Trainingsschnitt: nur Spiele VOR dem WM-Eroeffnungsspiel fliessen ins
# Training und in die Form ein - so bleibt jede Prognose echt out-of-sample.
CUTOFF = GROUP_STAGE_START.strftime("%Y-%m-%d")


def _ensure_sources(refresh: bool) -> None:
    """Echte Quelldaten sicherstellen (FC-26-Spieler + martj42-Historie)."""
    DATA.mkdir(parents=True, exist_ok=True)
    if refresh or not FC26_CSV.exists():
        print("Lade echte Spielerdaten (EA FC 26) ...")
        fetch_fifa_players(FC26_CSV)
    if refresh or not HISTORY_CSV.exists():
        print("Lade echte Historie (martj42/international_results) ...")
        fetch_international_results(HISTORY_CSV)


def _build_database(fixtures: pd.DataFrame) -> pd.DataFrame:
    """DB neu aufbauen: echte Historie + echte WM-2026-Fixtures."""
    con = connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS matches")
    cur.execute("DROP TABLE IF EXISTS fixtures")
    con.commit()
    init_schema(con)
    n_matches = load_matches(con, HISTORY_CSV)
    load_fixtures(con, FIXTURES_OUT)
    matches = read_table(con, "matches")
    con.close()
    print(f"DB: {n_matches} echte Spiele + {len(fixtures)} WM-Fixtures -> {DB_PATH}")
    return matches


def _rate_players_table(squads: pd.DataFrame, tg: dict) -> pd.DataFrame:
    """Jeden Spieler bewerten -> flache Tabelle mit Gesamtrating."""
    rows = []
    for _, r in squads.iterrows():
        ps = rate_player(dict(r))
        rows.append({
            "group": tg.get(r["team"], ""),
            "team": r["team"],
            "player": r["player"],
            "club": r.get("club", ""),
            "league": r.get("league", ""),
            "position": ps.position,
            "role": ps.role,
            "age": int(r.get("age", 0) or 0),
            "ea_overall": round(float(r.get("overall", 0) or 0), 0),
            "skill": ps.skill,
            "form": ps.form,
            "experience": ps.experience,
            "overall": ps.overall,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["team", "overall"], ascending=[True, False]).reset_index(drop=True)


def _rate_teams_table(team_scores: dict, squads: pd.DataFrame, tg: dict) -> pd.DataFrame:
    """Team-Gesamtratings -> Tabelle inkl. Gruppe, Kadergroesse, Rang."""
    rows = []
    for team, s in team_scores.items():
        rows.append({
            "group": tg.get(team, ""),
            "team": team,
            "overall": s.overall,
            "best_xi": s.best_xi_score,
            "depth": s.depth_score,
            "chemistry": round(s.chemistry.overall, 1),
            "coach": s.coach_score,
            "n_players": int((squads["team"] == team).sum()),
        })
    df = pd.DataFrame(rows).sort_values("overall", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def _fixture_contexts(fixtures: pd.DataFrame) -> dict:
    """Kontext-Elo je Spiel: Reiseweg (Heimat->Spielort) + Ruhetage.

    Reiseweg wirkt immer (interkontinentale Teams reisen weiter, Gastgeber
    sind lokal im Vorteil); Ruhetage ergeben sich aus dem Spielplan-Abstand
    zum vorigen Spiel je Team (in der Gruppenphase meist symmetrisch ~0).
    """
    from datetime import datetime

    last: dict = {}
    ctx: dict = {}
    for fx in fixtures.sort_values("date").itertuples(index=False):
        home, away, group = fx.home_team, fx.away_team, fx.group_name
        d = datetime.strptime(fx.date, "%Y-%m-%d")
        hr = (d - last[home]).days if home in last else None
        ar = (d - last[away]).days if away in last else None
        ce = rest_delta(hr, ar)
        ce += travel_delta(TEAM_COORD.get(home), TEAM_COORD.get(away), venue_coord(group))
        ctx[int(fx.match_id)] = round(float(ce), 1)
        last[home] = d
        last[away] = d
    return ctx


def _fit_score_model(train: pd.DataFrame, overall: dict) -> LogisticRegression:
    """Score-Differenz (+ Heimfeld) -> 1X2 (fuer das Ensemble), auf Historie."""
    rows, ys = [], []
    for r in train.itertuples(index=False):
        oh, oa = overall.get(r.home_team), overall.get(r.away_team)
        if oh is None or oa is None:
            continue
        hf = 0 if int(getattr(r, "neutral", 0) or 0) else 1
        rows.append([oh - oa, hf])
        ys.append(2 if r.home_score > r.away_score else (0 if r.away_score > r.home_score else 1))
    clf = LogisticRegression(max_iter=2000).fit(np.asarray(rows, float), np.asarray(ys, int))
    return clf


def _score_probs(clf: LogisticRegression, score_diff: float, neutral: int):
    """(p_home, p_draw, p_away) aus dem Score-Modell."""
    proba = clf.predict_proba(np.array([[score_diff, 0 if neutral else 1]], float))[0]
    out = {int(c): proba[i] for i, c in enumerate(clf.classes_)}
    return out.get(2, 0.0), out.get(1, 0.0), out.get(0, 0.0)


def _predict_fixtures(predictor, fixtures, hist, tg, contexts, score_clf, overall) -> pd.DataFrame:
    """Alle Gruppenspiele vorhersagen: Hybrid (mit Kontext) + Ensemble.

    - ``p_*``     : Hybrid (Elo + Kader + Tor-Modell + Logit) inkl. Reise/Ruhe.
    - ``ens_*``   : Ensemble aus Hybrid und reinem Score-Modell (im Backtest
                    bester Log-Loss/Brier - beste Wahrscheinlichkeits-Qualitaet).
    """
    rows = []
    for _, fx in fixtures.iterrows():
        ce = contexts.get(int(fx["match_id"]), 0.0)
        pred = predictor.predict_fixture(fx.to_dict(), hist, context_elo=ce)
        sd = overall.get(fx["home_team"], 0.0) - overall.get(fx["away_team"], 0.0)
        sh, sx, sa = _score_probs(score_clf, sd, int(fx["neutral"]))
        eh, ex, ea = 0.5 * pred.home_win + 0.5 * sh, 0.5 * pred.draw + 0.5 * sx, 0.5 * pred.away_win + 0.5 * sa
        probs = {"1": pred.home_win, "X": pred.draw, "2": pred.away_win}
        ens = {"1": eh, "X": ex, "2": ea}
        rows.append({
            "date": fx["date"],
            "group": fx["group_name"],
            "home_team": fx["home_team"],
            "away_team": fx["away_team"],
            "p_home": round(pred.home_win, 3),
            "p_draw": round(pred.draw, 3),
            "p_away": round(pred.away_win, 3),
            "xg_home": round(pred.expected_home_goals, 2),
            "xg_away": round(pred.expected_away_goals, 2),
            "tip": max(probs, key=probs.get),
            "ens_home": round(eh, 3),
            "ens_draw": round(ex, 3),
            "ens_away": round(ea, 3),
            "ens_tip": max(ens, key=ens.get),
            "context_elo": ce,
            "neutral": int(fx["neutral"]),
        })
    return pd.DataFrame(rows)


def _simulate_groups(predictor: WMPredictor, fixtures: pd.DataFrame, hist: pd.DataFrame, n: int) -> pd.DataFrame:
    """Weiterkommens-Wahrscheinlichkeiten je Gruppe (Monte Carlo)."""
    parts = []
    for group in GROUPS:
        gfx = fixtures[fixtures["group_name"] == group]
        sim = simulate_group(gfx, hist, predictor, n=n)
        sim.insert(0, "group", group)
        parts.append(sim)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="WM-2026-Pipeline mit echten Daten")
    parser.add_argument("--refresh", action="store_true", help="Quelldaten neu herunterladen")
    parser.add_argument("--sims", type=int, default=2000, help="Monte-Carlo-Durchlaeufe je Gruppe")
    parser.add_argument("--coverage", action="store_true",
                        help="Kader-Pull mit Datenabdeckung skalieren (duenne Kader vorsichtiger)")
    args = parser.parse_args()

    _ensure_sources(args.refresh)
    tg = team_group()
    teams = all_teams()

    print("=" * 78)
    print("WM 2026 - Aufbau mit echten Daten")
    print("=" * 78)

    # --- Schritt 1: echte Spieler -> Kader je Nation -----------------------
    players = load_fifa_as_squads(FC26_CSV, prefer_long_name=True)
    squads = build_squads(players, teams)
    squads.to_csv(SQUADS_OUT, index=False)
    print(f"[1] Kader gebaut: {len(squads)} Spieler aus EA-FC-26-Echtdaten, "
          f"{squads['team'].nunique()}/48 Nationen -> {SQUADS_OUT.name}")

    # --- Schritt 2: Spieler- und Team-Ratings ------------------------------
    fixtures = write_fixtures(FIXTURES_OUT)
    player_tbl = _rate_players_table(squads, tg)
    player_tbl.to_csv(PLAYERS_OUT, index=False)

    team_scores = rate_all_teams(squads)
    overall = {t: s.overall for t, s in team_scores.items()}
    coverage = coverage_confidence(squad_counts(squads))
    team_tbl = _rate_teams_table(team_scores, squads, tg)
    team_tbl.to_csv(TEAMS_OUT, index=False)
    print(f"[2] Spieler bewertet -> {PLAYERS_OUT.name}; Teams bewertet -> {TEAMS_OUT.name}")

    # --- Schritt 3: echte Historie -> Modell (leak-frei) + Kaderstaerke ----
    matches = _build_database(fixtures)
    train = matches[matches["date"] < CUTOFF].reset_index(drop=True)
    print(f"[3] Training auf {len(train)} echten Spielen VOR {CUTOFF} (out-of-sample) ...")
    predictor = WMPredictor(squad_pull=DEFAULT_SQUAD_PULL).fit(train)
    predictor.attach_team_scores(team_scores, coverage=coverage if args.coverage else None)
    predictor.save(MODEL_PATH)

    # --- Schritt 4: Vorhersagen (+ Reise/Ruhe + Ensemble) + Simulation -----
    contexts = _fixture_contexts(fixtures)
    score_clf = _fit_score_model(train, overall)
    preds = _predict_fixtures(predictor, fixtures, train, tg, contexts, score_clf, overall)
    preds.to_csv(PRED_OUT, index=False)
    sim = _simulate_groups(predictor, fixtures, train, args.sims)
    sim.to_csv(SIM_OUT, index=False)
    print(f"[4] {len(preds)} Gruppenspiele vorhergesagt -> {PRED_OUT.name}; "
          f"Gruppensimulation ({args.sims}x) -> {SIM_OUT.name}")

    # --- Schritt 5: komplette K.-o.-Phase (Titelchancen + Turnierbaum) -----
    from src import knockout as ko

    prepared = ko.prepare_tournament(predictor, train, GROUPS)
    tour = ko.simulate_tournament(predictor, train, GROUPS, fixtures=fixtures, n=args.sims, prepared=prepared)
    tour.probs.to_csv(KNOCKOUT_OUT, index=False)
    bracket, champion = ko.most_likely_bracket(predictor, train, GROUPS, fixtures=fixtures, prepared=prepared)
    ko.bracket_to_frame(bracket).to_csv(BRACKET_OUT, index=False)
    print(f"[5] Turnier simuliert ({args.sims}x) -> {KNOCKOUT_OUT.name}; "
          f"Turnierbaum bis zum Finale -> {BRACKET_OUT.name} (Weltmeister-Prognose: {champion})")

    _print_summary(player_tbl, team_tbl, preds, sim)


def _print_summary(players: pd.DataFrame, teams: pd.DataFrame, preds: pd.DataFrame, sim: pd.DataFrame) -> None:
    """Kompakte, lesbare Zusammenfassung in die Konsole."""
    pd.set_option("display.width", 120)

    print("\n" + "=" * 78)
    print("TOP-15 SPIELER (Gesamtrating)")
    print("=" * 78)
    top = players.sort_values("overall", ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"  {r['overall']:5.1f}  {r['player']:<26} {r['team']:<16} {r['position']:<3} {r['club']}")

    print("\n" + "=" * 78)
    print("TEAM-RANGLISTE (Gesamtrating)")
    print("=" * 78)
    for _, r in teams.iterrows():
        thin = "  (duenne Datenlage)" if r["n_players"] < 16 else ""
        print(f"  {r['rank']:2d}. [{r['group']}] {r['team']:<18} {r['overall']:5.1f}   "
              f"XI {r['best_xi']:.1f} | Tiefe {r['depth']:.1f} | Chemie {r['chemistry']:.1f} | "
              f"n={r['n_players']:2d}{thin}")

    print("\n" + "=" * 78)
    print("WEITERKOMMEN (P Top-2 je Gruppe, Monte Carlo)")
    print("=" * 78)
    for group in GROUPS:
        g = sim[sim["group"] == group].sort_values("P(Top 2)", ascending=False)
        line = "   ".join(f"{r['Team']} {r['P(Top 2)']*100:.0f}%" for _, r in g.iterrows())
        print(f"  Gruppe {group}: {line}")

    print("\n" + "=" * 78)
    print("BEISPIEL-VORHERSAGEN (Top-Spiele)")
    print("=" * 78)
    show = ["France", "Brazil", "Spain", "England", "Germany", "Argentina", "Portugal"]
    sample = preds[preds["home_team"].isin(show) | preds["away_team"].isin(show)].head(12)
    for _, r in sample.iterrows():
        print(f"  [{r['group']}] {r['home_team']:<16} vs {r['away_team']:<16}  "
              f"1={r['p_home']*100:4.1f}%  X={r['p_draw']*100:4.1f}%  2={r['p_away']*100:4.1f}%  "
              f"| xG {r['xg_home']:.2f}:{r['xg_away']:.2f}  -> {r['tip']}")

    print("\nFertig. Dashboard mit denselben Daten: `streamlit run app.py`")


if __name__ == "__main__":
    main()
