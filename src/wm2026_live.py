"""Live-Stand der laufenden WM 2026: tatsaechliche Spielergebnisse.

Einzige Quelle der **echten Endergebnisse** der bereits gespielten
Gruppenspiele. Beide Verbraucher teilen sich diese Daten:

- ``scripts/update_results.py`` schreibt daraus die Ergebnis-Tabelle
  (``data/wm2026_results.csv``) und gibt sie aus.
- ``scripts/build_wm2026.py`` arbeitet sie in die **Teamstaerken** ein:
  das Elo wird mit den gespielten Spielen fortgeschrieben und die Form-Historie
  ergaenzt, sodass die restlichen Spiele mit aktualisierten Staerken
  vorhergesagt werden.

Aktualisieren: neue Eintraege in ``ACTUAL_RESULTS`` ergaenzen (Schluessel =
``match_id`` aus ``data/wm2026_fixtures.csv``, Wert = ``(heim_tore, gast_tore)``)
und ``AS_OF`` hochsetzen.

Quellen (Web-Recherche, Stand 16.06.2026): CBS Sports, Yahoo Sports, SBS News,
Sky Sports, ESPN, Wikipedia (2026 FIFA World Cup group stage).
"""

from __future__ import annotations

import pandas as pd

# Datum, bis zu dem die Ergebnisse vollstaendig sind.
AS_OF = "2026-06-16"

# Tatsaechliche Endergebnisse: match_id -> (heim_tore, gast_tore).
# Orientierung (Heim/Gast) entspricht dem Spielplan in wm2026_fixtures.csv.
# Bisher gespielt: 1. Spieltag der Gruppen A-H (16 Spiele). Die Partien vom
# 16.06. (Gruppen I-L) stossen erst am Abend (ET) an und fehlen daher noch.
ACTUAL_RESULTS: dict[int, tuple[int, int]] = {
    # Gruppe A
    1: (2, 0),   # Mexico 2-0 South Africa
    2: (2, 1),   # South Korea 2-1 Czech Republic
    # Gruppe B
    7: (1, 1),   # Canada 1-1 Bosnia-Herzegovina
    8: (1, 1),   # Qatar 1-1 Switzerland
    # Gruppe C
    13: (1, 1),  # Brazil 1-1 Morocco
    14: (0, 1),  # Haiti 0-1 Scotland
    # Gruppe D
    19: (4, 1),  # United States 4-1 Paraguay
    20: (2, 0),  # Australia 2-0 Turkey
    # Gruppe E
    25: (7, 1),  # Germany 7-1 Curacao
    26: (1, 0),  # Ivory Coast 1-0 Ecuador
    # Gruppe F
    31: (2, 2),  # Netherlands 2-2 Japan
    32: (5, 1),  # Sweden 5-1 Tunisia
    # Gruppe G
    37: (1, 1),  # Belgium 1-1 Egypt
    38: (2, 2),  # Iran 2-2 New Zealand
    # Gruppe H
    43: (0, 0),  # Spain 0-0 Cape Verde
    44: (1, 1),  # Saudi Arabia 1-1 Uruguay
}

# K-Faktor-Wettbewerb fuer die Elo-Fortschreibung der gespielten Spiele.
TOURNAMENT = "FIFA World Cup"


def outcome(home_score: int, away_score: int) -> str:
    """1 = Heimsieg, X = Unentschieden, 2 = Auswaertssieg."""
    if home_score > away_score:
        return "1"
    if home_score < away_score:
        return "2"
    return "X"


def played_matches(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Gespielte WM-Spiele als matches-DataFrame (fuer Elo + Form).

    Verknuepft ``ACTUAL_RESULTS`` mit dem Spielplan und liefert die Spalten,
    die ``EloModel.update`` und das Form-Feature erwarten (inkl. ``tournament``
    fuer den korrekten K-Faktor). Reihenfolge chronologisch.
    """
    rows = []
    for _, m in fixtures.iterrows():
        score = ACTUAL_RESULTS.get(int(m["match_id"]))
        if score is None:
            continue
        hs, as_ = score
        rows.append({
            "date": m["date"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_score": int(hs),
            "away_score": int(as_),
            "neutral": int(m.get("neutral", 0) or 0),
            "tournament": TOURNAMENT,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("date").reset_index(drop=True)
    return df


def known_scores(fixtures: pd.DataFrame) -> dict[tuple[str, str], tuple[int, int]]:
    """Gespielte Ergebnisse als ``{(heim, gast): (heim_tore, gast_tore)}``.

    Fuer die Gruppensimulation, damit bereits gespielte Partien fixiert statt
    gesampelt werden.
    """
    from .teams import normalize_team

    out: dict[tuple[str, str], tuple[int, int]] = {}
    for _, m in fixtures.iterrows():
        score = ACTUAL_RESULTS.get(int(m["match_id"]))
        if score is None:
            continue
        key = (normalize_team(m["home_team"]), normalize_team(m["away_team"]))
        out[key] = (int(score[0]), int(score[1]))
    return out
