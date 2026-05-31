"""Erzeugung von Demo-Kaderdaten (deterministisch).

WICHTIG: Diese Spielerratings sind eine **Naeherung/Schaetzung**, keine
offiziellen Werte. Sie geben jedem WM-Team einen positionsbewussten Kader
(GK/DEF/MID/FWD) mit plausiblen Staerken rund um ein nationenspezifisches
Niveau, damit der Spieler->Team-Ansatz sofort demonstrierbar ist.

Fuer echte Prognosen eigene Kader-CSV einspeisen
(``--squads pfad.csv``), Format:
    team, player, position, rating[, available]

Die Spielernamen sind generisch (``<Land> Player k (POS)``) - hier zaehlt
ausschliesslich die *Struktur* (Niveau + Positionsverteilung), nicht die
Identitaet einzelner Spieler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Naeherungs-Teamniveau (mittleres Spielerrating der Stammelf), bewusst grob.
TEAM_LEVEL: Dict[str, float] = {
    "France": 85, "Spain": 85, "England": 84, "Brazil": 84, "Argentina": 85,
    "Portugal": 84, "Netherlands": 83, "Germany": 83, "Belgium": 82,
    "Italy": 82, "Croatia": 81, "Uruguay": 80, "Colombia": 80, "Morocco": 80,
    "Switzerland": 79, "Japan": 79, "Denmark": 79, "Mexico": 78, "Senegal": 79,
    "United States": 78, "Norway": 79, "Austria": 78, "Turkey": 78,
    "Ecuador": 77, "Serbia": 78, "South Korea": 78, "Ukraine": 77,
    "Sweden": 77, "Poland": 77, "Nigeria": 77, "Iran": 76, "Algeria": 77,
    "Chile": 76, "Egypt": 77, "Scotland": 76, "Czech Republic": 76,
    "Greece": 76, "Wales": 76, "Hungary": 76, "Ivory Coast": 77,
    "Peru": 75, "Canada": 76, "Australia": 75, "Cameroon": 76, "Mali": 76,
    "Paraguay": 75, "Ghana": 76, "Tunisia": 75, "Qatar": 74, "Venezuela": 75,
    "Romania": 74, "Ireland": 75, "Saudi Arabia": 73, "Burkina Faso": 74,
    "DR Congo": 74, "Costa Rica": 74, "Iraq": 72, "Uzbekistan": 73,
    "South Africa": 73, "Panama": 73, "Honduras": 72, "Jordan": 72,
    "Jamaica": 73, "Cape Verde": 72, "New Zealand": 71, "Curacao": 70,
    "Bolivia": 71,
}

# Positionsbezogener Offset auf das Teamniveau (Stuermer/Verteidiger einer
# Top-Nation sind im Schnitt leicht unterschiedlich stark verteilt).
POS_OFFSET = {"GK": -1.0, "DEF": -0.5, "MID": 0.5, "FWD": 0.5}
# Wie viele Spieler je Position erzeugt werden (Kader von 23).
POS_COUNT = {"GK": 3, "DEF": 8, "MID": 7, "FWD": 5}


def make_squads(seed: int = 13) -> pd.DataFrame:
    """Erzeugt einen 23er-Kader je Team mit positionsbewussten Ratings."""
    rng = np.random.default_rng(seed)
    rows = []
    for team, level in TEAM_LEVEL.items():
        for pos, count in POS_COUNT.items():
            base = level + POS_OFFSET[pos]
            # Stammspieler nahe am Niveau, Reservisten etwas darunter.
            for k in range(count):
                drop = 0.0 if k == 0 else rng.uniform(0.5, 5.5) * (k / count)
                rating = float(np.clip(base - drop + rng.normal(0, 1.0), 50, 95))
                rows.append(
                    {
                        "team": team,
                        "player": f"{team} Player {pos}{k + 1}",
                        "position": pos,
                        "rating": round(rating, 1),
                        "available": 1,
                    }
                )
    return pd.DataFrame(rows)


def write_sample_squads(data_dir) -> Path:
    """Schreibt die Demo-Kaderdaten als CSV und gibt den Pfad zurueck."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "sample_squads.csv"
    make_squads().to_csv(path, index=False)
    return path
