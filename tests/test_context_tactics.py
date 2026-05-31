"""Tests fuer Kontext- (Reise/Pause/Hoehe/Klima) und Taktik-Faktoren.

Kein Netzwerk noetig. Aufruf: ``python tests/test_context_tactics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.context import (
    CAP_TOTAL,
    altitude_delta,
    climate_delta,
    compute_context,
    rest_delta,
    travel_delta,
)
from src.rating.tactics import CAP_TACTICS, TacticProfile, matchup_delta


def test_rest_advantage_sign_and_cap():
    """Mehr Ruhe = positiver Heimvorteil; Wert bleibt gedeckelt."""
    assert rest_delta(7, 2) > 0
    assert rest_delta(2, 7) < 0
    assert abs(rest_delta(14, 0)) <= 25.0 + 1e-6
    assert rest_delta(None, 3) == 0.0  # fehlende Daten -> kein Effekt


def test_travel_penalizes_far_guest():
    """Weiter gereister Gast -> Heimvorteil (positiv)."""
    home = (52.5, 13.4)   # Berlin
    venue = (52.5, 13.4)  # Spiel in Berlin
    far_away = (-34.6, -58.4)  # Buenos Aires
    d = travel_delta(home, far_away, venue)
    assert d > 0
    assert d <= 30.0 + 1e-6


def test_altitude_only_high_venues():
    """Hoehen-Effekt erst ab ~1200 m; hoehengewohntes Heimteam im Vorteil."""
    assert altitude_delta(300, 100, 100) == 0.0           # Flachland: kein Effekt
    d = altitude_delta(2800, 2500, 50)                     # Heim hoehennah, Gast Flachland
    assert d > 0
    assert d <= 40.0 + 1e-6


def test_climate_needs_contrast():
    """Klima wirkt nur bei Hitze UND unterschiedlicher Klima-Herkunft."""
    assert climate_delta(20, 50, "tropical", "temperate") == 0.0  # nicht heiss
    hot = climate_delta(36, 70, "tropical", "temperate")
    assert hot > 0  # warmes Team profitiert
    assert climate_delta(36, 70, "tropical", "hot") == 0.0  # kein Kontrast


def test_context_total_capped():
    """Die Summe aller Kontextfaktoren bleibt innerhalb des Gesamt-Caps."""
    home = {"rest_days": 14, "coord": (19.4, -99.1), "altitude_m": 2240, "climate": "warm"}
    away = {"rest_days": 1, "coord": (35.7, 139.7), "altitude_m": 10, "climate": "temperate"}
    venue = {"coord": (19.4, -99.1), "altitude_m": 2240, "temp_c": 34, "humidity": 70}
    r = compute_context(home, away, venue)
    assert abs(r.elo_delta) <= CAP_TOTAL + 1e-6
    assert set(r.parts) == {"rest", "travel", "altitude", "climate"}


def test_tactics_counter_beats_pressing():
    """Konterstarkes Team hat Vorteil gegen hochpressenden Gegner."""
    counter = TacticProfile(counter=90, tempo=85, pressing=40)
    pressing = TacticProfile(pressing=90, possession=75, counter=40)
    d = matchup_delta(counter, pressing)
    assert d > 0
    assert abs(d) <= CAP_TACTICS + 1e-6


def test_tactics_symmetry():
    """Vertauscht man Heim/Gast, kehrt sich das Vorzeichen (etwa) um."""
    a = TacticProfile(counter=85, pressing=45, possession=70)
    b = TacticProfile(pressing=88, possession=80, counter=50)
    assert abs(matchup_delta(a, b) + matchup_delta(b, a)) < 1e-6


def test_identical_profiles_neutral():
    """Gleiche Profile -> kein Taktikvorteil."""
    p = TacticProfile(pressing=70, possession=70, tempo=70, block=70, width=70, counter=70)
    assert abs(matchup_delta(p, p)) < 1e-6


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
