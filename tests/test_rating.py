"""Tests fuer das Bewertungssystem (Kriterien, Spieler, Chemie, Team).

Kein Netzwerk, keine Trainingsdaten noetig - prueft die Logik direkt.
Aufruf: ``python -m pytest tests/test_rating.py`` oder ``python tests/test_rating.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.rating.criteria import (
    ROLE_PROFILES,
    ChemistryWeights,
    PlayerWeights,
    TeamWeights,
    league_strength,
)
from src.rating.player import rate_player
from src.rating.sample import COACHES, load_sample_players
from src.rating.team import rate_all_teams, rate_team


def test_weights_sum_to_one():
    """Alle Gewichtsklassen summieren (annaehernd) auf 1.0."""
    for w in (PlayerWeights(), TeamWeights(), ChemistryWeights()):
        total = sum(w.as_dict().values())
        assert abs(total - 1.0) < 1e-6, f"{w} summiert auf {total}"


def test_role_profiles_sum_to_one():
    """Jedes Positionsprofil summiert auf 1.0 (saubere Gewichtung)."""
    for role, profile in ROLE_PROFILES.items():
        total = sum(profile.values())
        assert abs(total - 1.0) < 1e-6, f"Profil {role} summiert auf {total}"


def test_position_specific_skill():
    """Dieselben Attribute ergeben je nach Position andere Skills."""
    base = {"team": "X", "player": "p", "pace": 90, "shooting": 90, "passing": 50,
            "dribbling": 80, "defending": 40, "physical": 70, "vision": 50, "gk": 10,
            "caps": 30, "age": 26}
    striker = rate_player({**base, "position": "ST"})
    cb = rate_player({**base, "position": "CB"})
    # Mit hohem Abschluss/Tempo, aber schwacher Defense: Stuermer-Skill > IV-Skill.
    assert striker.skill > cb.skill


def test_striker_beats_weak_forward():
    """Ein Topstuermer schlaegt einen schwachen Stuermer klar."""
    top = rate_player({"team": "A", "player": "top", "position": "ST", "pace": 95,
                       "shooting": 95, "dribbling": 85, "physical": 80, "vision": 80,
                       "passing": 70, "defending": 40, "gk": 10, "caps": 70, "age": 27})
    weak = rate_player({"team": "B", "player": "weak", "position": "ST", "pace": 70,
                        "shooting": 68, "dribbling": 64, "physical": 70, "vision": 66,
                        "passing": 60, "defending": 40, "gk": 10, "caps": 20, "age": 25})
    assert top.overall > weak.overall + 8


def test_league_strength_ordering():
    """Top-Liga staerker als schwaechere/Default-Liga."""
    assert league_strength("Real Madrid") > league_strength("Al-Hilal")
    assert league_strength(None, "Premier League") > league_strength(None, "Unbekannt")


def test_chemistry_does_not_inflate_weak_squad():
    """Hohe Chemie hebt einen schwachen Kader nicht ueber einen starken."""
    players = load_sample_players()
    teams = rate_all_teams(players, coaches=COACHES)
    # Trotz hoher Saudi-Chemie bleibt Argentinien klar staerker.
    assert teams["Argentina"].overall > teams["Saudi Arabia"].overall + 6
    # Chemie-Effekt eines schwachen Kaders darf den Gesamtscore nicht
    # ueber sein reines Niveau hinaus stark anheben.
    saudi = teams["Saudi Arabia"]
    assert saudi.overall <= saudi.best_xi_score + 3


def test_scores_in_range():
    """Alle Team-Scores liegen in [0,100]."""
    teams = rate_all_teams(load_sample_players(), coaches=COACHES)
    for t in teams.values():
        assert 0 <= t.overall <= 100
        assert 0 <= t.chemistry.overall <= 100


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [OK ] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  [XX ] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} Tests bestanden.")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
