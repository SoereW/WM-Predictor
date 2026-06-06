"""Erststart-Vorbereitung: DB + Modell aus den eingecheckten Echtdaten bauen.

Bewusst **ohne** Netzwerk: nutzt die im Repo liegenden Daten
(``international_results.csv`` als Historie, ``wm2026_squads.csv`` als Kader).
So laeuft die App auch offline sofort. Ein kompletter Neuaufbau inkl. frischer
Online-Daten geht weiter ueber ``scripts/build_wm2026.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .database import connect, init_schema, load_fixtures, load_matches, read_table
from .model import WMPredictor
from .rating.squad_builder import coverage_confidence, squad_counts
from .rating.team import rate_all_teams
from .wm2026 import GROUP_STAGE_START

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
MODEL_PATH = ROOT / "models" / "wm_predictor.joblib"
SQUADS_CSV = DATA / "wm2026_squads.csv"
DEFAULT_SQUAD_PULL = 0.50
CUTOFF = GROUP_STAGE_START.strftime("%Y-%m-%d")


def _build_db() -> None:
    real = DATA / "international_results.csv"
    sample = DATA / "sample_matches.csv"
    wm_fixtures = DATA / "wm2026_fixtures.csv"
    sample_fixtures = DATA / "sample_fixtures_2026.csv"
    if not real.exists() and not sample.exists():
        from .sample_data import write_sample_data
        write_sample_data(DATA)
    if not wm_fixtures.exists() and not sample_fixtures.exists():
        from .sample_data import write_sample_data
        write_sample_data(DATA)
    fixtures = wm_fixtures if wm_fixtures.exists() else sample_fixtures

    con = connect(DB_PATH)
    init_schema(con)
    load_matches(con, real if real.exists() else sample)
    load_fixtures(con, fixtures)
    con.close()


def _build_model() -> None:
    con = connect(DB_PATH)
    matches = read_table(con, "matches")
    con.close()
    train = matches[matches["date"] < CUTOFF]
    train = train.reset_index(drop=True) if len(train) else matches.reset_index(drop=True)
    predictor = WMPredictor(squad_pull=DEFAULT_SQUAD_PULL).fit(train)
    try:
        if SQUADS_CSV.exists():
            squads = pd.read_csv(SQUADS_CSV)
            predictor.attach_team_scores(rate_all_teams(squads),
                                         coverage=coverage_confidence(squad_counts(squads)))
    except Exception:
        pass
    predictor.save(MODEL_PATH)


def ensure_assets(verbose: bool = False) -> None:
    """Stellt sicher, dass DB und Modell existieren (baut sie bei Bedarf)."""
    if not DB_PATH.exists():
        if verbose:
            print("• Lege Datenbank aus den eingecheckten Echtdaten an ...")
        _build_db()
    if not MODEL_PATH.exists():
        if verbose:
            print("• Trainiere das Modell (einmalig, ~1 Min.) ...")
        _build_model()
    if verbose:
        print("• Bereit.")
