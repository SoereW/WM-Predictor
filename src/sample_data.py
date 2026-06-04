"""Erzeugung der mitgelieferten Beispieldaten (deterministisch).

WICHTIG: Die historischen Spiele sind **synthetisch** (seed-basiert aus
latenten Teamstaerken via Poisson-Tore erzeugt) - keine echten Resultate.
Sie geben dem Modell eine realistische Struktur (staerkere Teams gewinnen
oefter), damit das Dashboard sofort sinnvoll laeuft. Fuer echte Prognosen:
``python scripts/init_db.py --fetch`` (laedt echte Historie von martj42).

Weil alles deterministisch aus Code entsteht, werden die CSVs nicht ins
Repo eingecheckt, sondern bei Bedarf erzeugt (`write_sample_data`); sowohl
das CLI-Skript als auch das Dashboard nutzen diese Funktionen.

Die Fixtures bilden das WM-2026-Format ab: 12 Gruppen (A-L) mit je 4
Teams und vollstaendigem Rundenturnier (6 Spiele/Gruppe). Die konkrete
Gruppeneinteilung ist ein plausibles Beispiel und sollte vor echtem
Einsatz durch die offizielle Auslosung ersetzt werden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Ungefaehre Staerken (Elo-aehnlich) - nur zur Datenerzeugung.
STRENGTH = {
    "Argentina": 2100, "France": 2050, "Spain": 2040, "England": 2010,
    "Brazil": 2000, "Portugal": 1990, "Netherlands": 1980, "Belgium": 1940,
    "Italy": 1930, "Germany": 1930, "Croatia": 1900, "Uruguay": 1900,
    "Colombia": 1880, "Morocco": 1870, "Switzerland": 1830, "Japan": 1830,
    "Denmark": 1820, "Mexico": 1820, "Senegal": 1820, "United States": 1800,
    "Norway": 1800, "Austria": 1790, "Turkey": 1780, "Ecuador": 1780,
    "Serbia": 1770, "South Korea": 1770, "Ukraine": 1760, "Sweden": 1760,
    "Poland": 1760, "Nigeria": 1760, "Iran": 1760, "Algeria": 1750,
    "Chile": 1740, "Egypt": 1740, "Scotland": 1740, "Czech Republic": 1740,
    "Greece": 1740, "Wales": 1740, "Hungary": 1760, "Ivory Coast": 1720,
    "Peru": 1720, "Canada": 1720, "Australia": 1720, "Cameroon": 1700,
    "Mali": 1700, "Paraguay": 1700, "Ghana": 1700, "Tunisia": 1690,
    "Qatar": 1680, "Venezuela": 1680, "Romania": 1680, "Ireland": 1680,
    "Bolivia": 1620, "Saudi Arabia": 1660, "Burkina Faso": 1660,
    "DR Congo": 1660, "Costa Rica": 1660, "Iraq": 1640, "Uzbekistan": 1640,
    "South Africa": 1640, "Panama": 1640, "Honduras": 1620, "Jordan": 1620,
    "Jamaica": 1620, "Cape Verde": 1620, "New Zealand": 1600, "Curacao": 1560,
}

# Beispiel-Gruppen (12 x 4). Bewusst ausgewogen, klar als Beispiel markiert.
GROUPS = {
    "A": ["Mexico", "Norway", "Ivory Coast", "Saudi Arabia"],
    "B": ["Canada", "Croatia", "Nigeria", "Qatar"],
    "C": ["United States", "Netherlands", "Egypt", "New Zealand"],
    "D": ["Argentina", "Sweden", "South Korea", "Panama"],
    "E": ["France", "Austria", "Japan", "Honduras"],
    "F": ["Brazil", "Switzerland", "Algeria", "Jordan"],
    "G": ["England", "Denmark", "Senegal", "Costa Rica"],
    "H": ["Spain", "Serbia", "Cameroon", "Uzbekistan"],
    "I": ["Portugal", "Turkey", "Ghana", "Iraq"],
    "J": ["Belgium", "Colombia", "Tunisia", "Australia"],
    "K": ["Germany", "Uruguay", "Morocco", "Jamaica"],
    "L": ["Italy", "Ecuador", "Iran", "Paraguay"],
}

HOST_CITY = {
    "A": "Mexico City", "B": "Toronto", "C": "New York", "D": "Dallas",
    "E": "Los Angeles", "F": "Miami", "G": "Atlanta", "H": "Houston",
    "I": "Seattle", "J": "Kansas City", "K": "Philadelphia", "L": "Vancouver",
}
HOSTS = {"United States", "Mexico", "Canada"}

TOURNAMENTS = [
    ("Friendly", 0.45),
    ("FIFA World Cup qualification", 0.30),
    ("UEFA Euro qualification", 0.12),
    ("Continental Cup", 0.13),
]


def _goals(rng, s_home, s_away, home_adv):
    diff = s_home - s_away + home_adv
    lh = np.clip(np.exp(np.log(1.35) + 0.0035 * diff), 0.12, 5.0)
    la = np.clip(np.exp(np.log(1.35) - 0.0035 * diff), 0.12, 5.0)
    return int(rng.poisson(lh)), int(rng.poisson(la))


def make_matches(n: int = 4200, seed: int = 7) -> pd.DataFrame:
    """Erzeugt synthetische, aber strukturierte historische Spiele."""
    rng = np.random.default_rng(seed)
    teams = list(STRENGTH.keys())
    names = np.array([t[0] for t in TOURNAMENTS])
    probs = np.array([t[1] for t in TOURNAMENTS])
    probs = probs / probs.sum()
    start = pd.Timestamp("2019-01-01")
    rows = []
    for _ in range(n):
        home, away = rng.choice(teams, size=2, replace=False)
        tour = rng.choice(names, p=probs)
        neutral = 1 if (tour == "Continental Cup" or rng.random() < 0.18) else 0
        home_adv = 0.0 if neutral else 60.0
        hs, as_ = _goals(rng, STRENGTH[home], STRENGTH[away], home_adv)
        day = start + pd.Timedelta(days=int(rng.integers(0, 365 * 7)))
        rows.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "tournament": tour,
                "city": "",
                "country": "",
                "neutral": neutral,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def make_fixtures() -> pd.DataFrame:
    """Erzeugt den Beispiel-Gruppenspielplan (12 Gruppen, je 6 Spiele)."""
    rounds = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
    base = pd.Timestamp("2026-06-11")
    rows = []
    mid = 1
    for gi, (g, teams) in enumerate(GROUPS.items()):
        for ri, (a, b) in enumerate(rounds):
            home, away = teams[a], teams[b]
            if away in HOSTS and home not in HOSTS:
                home, away = away, home
            neutral = 0 if home in HOSTS else 1
            day = base + pd.Timedelta(days=gi + ri * 4)
            rows.append(
                {
                    "match_id": mid,
                    "date": day.strftime("%Y-%m-%d"),
                    "home_team": home,
                    "away_team": away,
                    "city": HOST_CITY[g],
                    "country": "USA/Canada/Mexico",
                    "neutral": neutral,
                    "group_name": g,
                    "stage": "group",
                }
            )
            mid += 1
    return pd.DataFrame(rows)


def write_sample_data(data_dir) -> Tuple[Path, Path]:
    """Schreibt Beispiel-Matches und -Fixtures als CSV; gibt die Pfade zurueck."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    matches_path = data_dir / "sample_matches.csv"
    fixtures_path = data_dir / "sample_fixtures_2026.csv"
    make_matches().to_csv(matches_path, index=False)
    make_fixtures().to_csv(fixtures_path, index=False)
    return matches_path, fixtures_path
