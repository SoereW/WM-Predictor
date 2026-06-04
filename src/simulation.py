"""Monte-Carlo-Simulation von Gruppen/Turnieren.

Aus dem Korrektergebnis-Gitter jedes Spiels werden Scorelines gezogen,
Tabellen gerechnet und ueber viele Durchlaeufe Platzierungs-
Wahrscheinlichkeiten geschaetzt.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .teams import normalize_team


def _fixture_draws(predictor, matches, fixtures) -> List[tuple]:
    """Bereitet je Spiel das flache Wahrscheinlichkeitsgitter zum Sampling vor."""
    prepared = []
    for _, fx in fixtures.iterrows():
        m = predictor.score_matrix(fx.to_dict(), matches)
        prepared.append(
            (
                normalize_team(fx["home_team"]),
                normalize_team(fx["away_team"]),
                m.flatten(),
                m.shape[0],
            )
        )
    return prepared


def simulate_group(group_fixtures: pd.DataFrame, matches: pd.DataFrame, predictor, n: int = 500) -> pd.DataFrame:
    """Schaetzt Gruppen-Platzierungswahrscheinlichkeiten per Monte Carlo.

    Rueckgabe: DataFrame je Team mit P(Platz 1), P(Top 2) und mittleren
    Punkten/Tordifferenz, absteigend nach P(Top 2).
    """
    teams = sorted(
        set(group_fixtures["home_team"].map(normalize_team))
        | set(group_fixtures["away_team"].map(normalize_team))
    )
    prepared = _fixture_draws(predictor, matches, group_fixtures)
    rng = np.random.default_rng(42)

    first = {t: 0 for t in teams}
    top2 = {t: 0 for t in teams}
    sum_points = {t: 0.0 for t in teams}
    sum_gd = {t: 0.0 for t in teams}

    for _ in range(n):
        pts = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        gs = {t: 0 for t in teams}
        for home, away, flat, kp1 in prepared:
            idx = rng.choice(len(flat), p=flat)
            hg, ag = divmod(int(idx), kp1)
            gs[home] += hg
            gs[away] += ag
            gd[home] += hg - ag
            gd[away] += ag - hg
            if hg > ag:
                pts[home] += 3
            elif hg < ag:
                pts[away] += 3
            else:
                pts[home] += 1
                pts[away] += 1

        order = sorted(teams, key=lambda t: (pts[t], gd[t], gs[t]), reverse=True)
        first[order[0]] += 1
        top2[order[0]] += 1
        if len(order) > 1:
            top2[order[1]] += 1
        for t in teams:
            sum_points[t] += pts[t]
            sum_gd[t] += gd[t]

    rows = [
        {
            "Team": t,
            "P(Platz 1)": round(first[t] / n, 3),
            "P(Top 2)": round(top2[t] / n, 3),
            "Ø Punkte": round(sum_points[t] / n, 2),
            "Ø Tordiff": round(sum_gd[t] / n, 2),
        }
        for t in teams
    ]
    df = pd.DataFrame(rows).sort_values(["P(Top 2)", "P(Platz 1)"], ascending=False)
    return df.reset_index(drop=True)
