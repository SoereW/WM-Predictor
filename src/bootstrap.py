"""Erststart-Bootstrap: aus eingecheckten/echten Daten alles aufbauen.

Ein Aufruf von :func:`ensure_assets` stellt sicher, dass

1. die **echte** Laenderspiel-Historie vorliegt (Download von
   ``martj42/international_results`` beim ersten Mal; offline-Fallback auf
   deterministische Beispieldaten),
2. die SQLite-Datenbank (Historie + WM-2026-Spielplan) existiert,
3. ein **leak-frei vor WM-Beginn** trainiertes Modell mit eingekoppelter
   Kaderstaerke gespeichert ist, und
4. alle Prognose-Artefakte (Gruppenspiel-Vorhersagen, Gruppensimulation,
   Titelchancen und der Turnierbaum bis zum Finale) als CSV bereitliegen.

Damit laeuft das Dashboard mit **einem Befehl** los - fehlt etwas, wird es
automatisch gebaut; ist alles da, ist der Aufruf praktisch kostenlos.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wm2026 import GROUP_STAGE_START

DATA = ROOT / "data"
DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
MODEL_PATH = ROOT / "models" / "wm_predictor.joblib"

HISTORY_CSV = DATA / "international_results.csv"
FIXTURES_CSV = DATA / "wm2026_fixtures.csv"
SQUADS_CSV = DATA / "wm2026_squads.csv"
PRED_CSV = DATA / "wm2026_predictions.csv"
GROUP_SIM_CSV = DATA / "wm2026_group_sim.csv"
KNOCKOUT_CSV = DATA / "wm2026_knockout_probs.csv"
BRACKET_CSV = DATA / "wm2026_bracket.csv"

# Leak-frei: nur Spiele VOR dem WM-Eroeffnungsspiel fliessen ins Training.
CUTOFF = GROUP_STAGE_START.strftime("%Y-%m-%d")
# Standardgewicht der aktuellen Kaderstaerke. Bewusst hoch (70 %): der aktuelle
# Kader inkl. Chemie/Tiefe/Trainer dominiert die Prognose, die Historie/Elo
# korrigiert nur noch. Im Dashboard bis 0.9 regelbar.
DEFAULT_SQUAD_PULL = 0.7
# Monte-Carlo-Durchlaeufe fuer die vorgerechneten Standard-Artefakte.
DEFAULT_SIMS = 4000


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


# ---------------------------------------------------------------------------
# Einzelschritte.
# ---------------------------------------------------------------------------
def _ensure_history(verbose: bool) -> None:
    """Echte Historie sicherstellen (Download; Fallback Beispieldaten)."""
    DATA.mkdir(parents=True, exist_ok=True)
    if HISTORY_CSV.exists():
        return
    try:
        from src.ingest import fetch_international_results

        _log(verbose, "• Lade echte Laenderspiel-Historie (martj42, ~49k Spiele) ...")
        fetch_international_results(HISTORY_CSV)
        _log(verbose, f"  ✓ Historie gespeichert: {HISTORY_CSV.name}")
    except Exception as err:  # noqa: BLE001
        _log(verbose, f"  ! Download nicht moeglich ({err}); nutze Beispieldaten (offline).")
        from src.sample_data import write_sample_data

        real, _sample_fixtures = write_sample_data(DATA)
        # write_sample_data legt sample_matches.csv an -> als Historie nutzen.
        sample_matches = DATA / "sample_matches.csv"
        if sample_matches.exists() and not HISTORY_CSV.exists():
            HISTORY_CSV.write_bytes(sample_matches.read_bytes())


def _ensure_fixtures(verbose: bool) -> None:
    """Offiziellen Gruppenspielplan sicherstellen."""
    if FIXTURES_CSV.exists():
        return
    from src.wm2026 import write_fixtures

    _log(verbose, "• Erzeuge WM-2026-Gruppenspielplan ...")
    write_fixtures(FIXTURES_CSV)


def _ensure_database(verbose: bool) -> None:
    """SQLite-DB (Historie + Fixtures) aufbauen, falls nicht vorhanden."""
    if DB_PATH.exists():
        return
    from src.database import connect, init_schema, load_fixtures, load_matches

    _log(verbose, "• Baue Datenbank (Historie + Spielplan) ...")
    con = connect(DB_PATH)
    init_schema(con)
    n = load_matches(con, HISTORY_CSV)
    load_fixtures(con, FIXTURES_CSV)
    con.close()
    _log(verbose, f"  ✓ {n} historische Spiele + Spielplan -> {DB_PATH.relative_to(ROOT)}")


def _team_scores_and_coverage():
    """Team-Ratings + Datenabdeckung aus den (echten) Kaderdaten."""
    from src.rating.squad_builder import coverage_confidence, squad_counts
    from src.rating.team import rate_all_teams

    squads = pd.read_csv(SQUADS_CSV)
    scores = rate_all_teams(squads)
    coverage = coverage_confidence(squad_counts(squads))
    return scores, coverage


def _ensure_model(verbose: bool) -> None:
    """Modell leak-frei trainieren, Kaderstaerke einkoppeln, speichern."""
    if MODEL_PATH.exists():
        return
    from src.database import connect, read_table
    from src.model import WMPredictor

    _log(verbose, f"• Trainiere Modell (nur Spiele vor {CUTOFF}, leak-frei) ...")
    con = connect(DB_PATH)
    matches = read_table(con, "matches")
    con.close()
    train = matches[matches["date"] < CUTOFF]
    train = train.reset_index(drop=True) if len(train) else matches.reset_index(drop=True)

    predictor = WMPredictor(squad_pull=DEFAULT_SQUAD_PULL).fit(train)
    if SQUADS_CSV.exists():
        try:
            scores, coverage = _team_scores_and_coverage()
            predictor.attach_team_scores(scores, coverage=None)  # voller Kader-Pull (Standard)
            _log(verbose, f"  ✓ Kaderstaerke fuer {len(scores)} Teams eingekoppelt (Gewicht {DEFAULT_SQUAD_PULL:.0%})")
        except Exception as err:  # noqa: BLE001
            _log(verbose, f"  ! Kaderstaerke nicht eingekoppelt ({err}); reines Elo/ML-Modell.")
    predictor.save(MODEL_PATH)
    _log(verbose, f"  ✓ Modell gespeichert -> {MODEL_PATH.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Prognose-Artefakte (Vorhersagen, Gruppen, Titelchancen, Turnierbaum).
# ---------------------------------------------------------------------------
def _fixture_contexts(fixtures: pd.DataFrame) -> dict:
    """Kontext-Elo je Gruppenspiel: Reiseweg (Heimat->Spielort) + Ruhetage."""
    from src.context import rest_delta, travel_delta
    from src.teams import normalize_team
    from src.wm2026 import TEAM_COORD, fixture_venue_coord

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


def _predict_group_games(predictor, fixtures, prepared, contexts) -> pd.DataFrame:
    """Alle Gruppenspiele aus den vorbereiteten Zustaenden vorhersagen."""
    from src.teams import normalize_team

    from src import knockout as ko

    rows = []
    for _, fx in fixtures.iterrows():
        home, away = normalize_team(fx["home_team"]), normalize_team(fx["away_team"])
        ce = contexts.get(int(fx["match_id"]), 0.0)
        ph, pdr, pa, lh, la = predictor.matchup_from_states(home, away, int(fx["neutral"]), prepared.states, extra_elo=ce)
        probs = {"1": ph, "X": pdr, "2": pa}
        sh, sa = ko.argmax_scoreline(predictor._score_grid(lh, la))
        rows.append({
            "date": fx["date"], "group": fx["group_name"], "home_team": home, "away_team": away,
            "p_home": round(ph, 3), "p_draw": round(pdr, 3), "p_away": round(pa, 3),
            "xg_home": round(lh, 2), "xg_away": round(la, 2),
            "score_home": sh, "score_away": sa,
            "tip": max(probs, key=probs.get), "context_elo": ce, "neutral": int(fx["neutral"]),
        })
    return pd.DataFrame(rows)


def build_artifacts(predictor, history, fixtures, n_sims: int = DEFAULT_SIMS, verbose: bool = False) -> dict:
    """Erzeugt **alle** Prognose-Artefakte konsistent aus einem Modell.

    Schreibt Vorhersagen, Gruppensimulation, Titelchancen und Turnierbaum als
    CSV und gibt die DataFrames zurueck. Genutzt vom Bootstrap und von
    ``scripts/build_wm2026.py``.
    """
    from src import knockout as ko
    from src.wm2026 import GROUPS

    _log(verbose, "• Berechne Vorhersagen, Turnier-Simulation und Turnierbaum ...")
    prepared = ko.prepare_tournament(predictor, history, GROUPS)
    contexts = _fixture_contexts(fixtures)

    preds = _predict_group_games(predictor, fixtures, prepared, contexts)
    tour = ko.simulate_tournament(predictor, history, GROUPS, fixtures=fixtures, n=n_sims, prepared=prepared)
    bracket_rows, champion = ko.most_likely_bracket(predictor, history, GROUPS, fixtures=fixtures, prepared=prepared)
    bracket_df = ko.bracket_to_frame(bracket_rows)

    group_sim = tour.groups.rename(columns={"team": "Team"})[
        ["group", "Team", "P(Platz 1)", "P(Top 2)", "Ø Punkte", "Ø Tordiff"]
    ]

    preds.to_csv(PRED_CSV, index=False)
    group_sim.to_csv(GROUP_SIM_CSV, index=False)
    tour.probs.to_csv(KNOCKOUT_CSV, index=False)
    bracket_df.to_csv(BRACKET_CSV, index=False)
    _log(verbose, f"  ✓ Turnierbaum bis zum Finale: Weltmeister-Prognose **{champion}**")
    return {"predictions": preds, "group_sim": group_sim, "knockout": tour.probs,
            "bracket": bracket_df, "champion": champion}


def _ensure_artifacts(verbose: bool) -> None:
    """Prognose-Artefakte erzeugen, falls ein Standard-Artefakt fehlt."""
    needed = [PRED_CSV, GROUP_SIM_CSV, KNOCKOUT_CSV, BRACKET_CSV]
    if all(p.exists() for p in needed):
        return
    from src.database import connect, read_table
    from src.model import WMPredictor

    con = connect(DB_PATH)
    matches = read_table(con, "matches")
    con.close()
    history = matches[matches["date"] < CUTOFF]
    history = history.reset_index(drop=True) if len(history) else matches.reset_index(drop=True)
    fixtures = pd.read_csv(FIXTURES_CSV)
    predictor = WMPredictor.load(MODEL_PATH)
    build_artifacts(predictor, history, fixtures, n_sims=DEFAULT_SIMS, verbose=verbose)


# ---------------------------------------------------------------------------
# Oeffentlicher Einstieg.
# ---------------------------------------------------------------------------
def ensure_assets(verbose: bool = False) -> None:
    """Stellt Historie, DB, Modell und alle Prognose-Artefakte sicher."""
    _ensure_history(verbose)
    _ensure_fixtures(verbose)
    _ensure_database(verbose)
    _ensure_model(verbose)
    _ensure_artifacts(verbose)
    _log(verbose, "• Alles bereit.")


if __name__ == "__main__":
    ensure_assets(verbose=True)
