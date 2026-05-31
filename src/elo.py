"""World-Football-Elo.

Sequentielles Elo mit Tor-Differenz-Multiplikator, turnierabhaengigem
K-Faktor und Heimvorteil. Liefert sowohl die finalen Ratings als auch die
*Pre-Match*-Ratings je Spiel (wichtig, um Trainingsfeatures ohne
Information-Leakage zu bauen) und eine Zeitreihe je Team fuer
Stichtags-Abfragen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_RATING = 1500.0
HOME_ADVANTAGE = 65.0  # Elo-Punkte Heimvorteil (nur bei nicht-neutralem Platz)


def tournament_weight(name: object) -> float:
    """K-Faktor nach Wichtigkeit des Wettbewerbs (World-Football-Elo-Logik)."""
    n = str(name or "").lower()
    is_qual = "qualif" in n
    if "friendly" in n:
        return 20.0
    if "world cup" in n:
        return 40.0 if is_qual else 60.0
    if any(
        k in n
        for k in (
            "euro", "copa am", "african cup", "afcon", "asian cup",
            "gold cup", "nations league", "confederations", "copa america",
        )
    ):
        return 40.0 if is_qual else 50.0
    if is_qual:
        return 40.0
    return 30.0


def gd_multiplier(goal_diff: int) -> float:
    """Multiplikator fuer das Tor-Verhaeltnis (hoehere Siege zaehlen mehr)."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def expected_score(rating_diff: float) -> float:
    """Erwartetes Resultat (0..1) aus der Rating-Differenz."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


@dataclass
class EloModel:
    base_rating: float = BASE_RATING
    home_advantage: float = HOME_ADVANTAGE
    ratings: Dict[str, float] = field(default_factory=dict)
    last_date: Dict[str, str] = field(default_factory=dict)
    history: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    pre_home: np.ndarray = field(default_factory=lambda: np.array([]))
    pre_away: np.ndarray = field(default_factory=lambda: np.array([]))

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    def fit(self, matches: pd.DataFrame) -> "EloModel":
        """Berechnet Ratings sequentiell ueber alle Spiele (chronologisch)."""
        df = matches.sort_values("date").reset_index(drop=True)
        pre_home = np.empty(len(df), dtype=float)
        pre_away = np.empty(len(df), dtype=float)

        for i, row in enumerate(df.itertuples(index=False)):
            home, away = row.home_team, row.away_team
            r_home, r_away = self._get(home), self._get(away)
            pre_home[i] = r_home
            pre_away[i] = r_away

            neutral = int(getattr(row, "neutral", 0) or 0)
            hfa = 0.0 if neutral else self.home_advantage
            exp_home = expected_score(r_home + hfa - r_away)

            hs, as_ = int(row.home_score), int(row.away_score)
            if hs > as_:
                actual_home = 1.0
            elif hs == as_:
                actual_home = 0.5
            else:
                actual_home = 0.0

            k = tournament_weight(getattr(row, "tournament", None))
            change = k * gd_multiplier(hs - as_) * (actual_home - exp_home)

            self.ratings[home] = r_home + change
            self.ratings[away] = r_away - change
            self.last_date[home] = row.date
            self.last_date[away] = row.date
            self.history.setdefault(home, []).append((row.date, self.ratings[home]))
            self.history.setdefault(away, []).append((row.date, self.ratings[away]))

        self.pre_home = pre_home
        self.pre_away = pre_away
        return self

    def current_rating(self, team: str) -> float:
        """Aktuellstes Rating eines Teams (oder Basis-Rating fuer Unbekannte)."""
        return self._get(team)

    def rating_asof(self, team: str, date: str) -> float:
        """Rating eines Teams unmittelbar vor einem Stichtag (kein Leakage)."""
        hist = self.history.get(team)
        if not hist:
            return self.base_rating
        rating = self.base_rating
        for d, r in hist:
            if d < date:
                rating = r
            else:
                break
        return rating
