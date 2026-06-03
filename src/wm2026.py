"""WM-2026-Turnierdefinition: echte Gruppen-Auslosung + Spielplan-Generator.

Die Gruppen entsprechen der offiziellen Final-Auslosung vom 5. Dezember 2025
(Washington, D.C.) inklusive der vier europaeischen Playoff-Sieger vom
31. Maerz 2026 (Bosnien-Herzegowina, Tschechien, Schweden, Tuerkei).

Teamnamen sind bereits in der **kanonischen** Schreibweise des Projekts
(siehe ``src/teams.normalize_team``), damit sie ohne weitere Umbenennung mit
der historischen Elo-/Form-Berechnung und den FC-Spielerdaten
zusammenpassen (z. B. "Korea Republic" -> "South Korea", "Czechia" ->
"Czech Republic", "Tuerkiye" -> "Turkey", "Cabo Verde" -> "Cape Verde").

> Hinweis: Auslosung per Web-Recherche zusammengetragen; vor produktivem
> Einsatz gegen die offizielle FIFA-Quelle abgleichen. Eine Aenderung hier
> reicht aus, der Rest des Spielplans wird daraus generiert.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import pandas as pd

# Offizielle Gruppen (kanonische Teamnamen).
GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia-Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Gastgeber (geniessen Heimvorteil = Spiel nicht "neutral", wenn sie als
# Heimteam gefuehrt werden).
HOSTS = {"United States", "Canada", "Mexico"}

# Eroeffnung der Gruppenphase (11.06.2026). Exakte Termine sind fuer die
# Vorhersage unkritisch (das Modell nutzt die finale Elo/Form), dienen aber
# der sauberen Sortierung/Anzeige im Dashboard.
GROUP_STAGE_START = date(2026, 6, 11)

# Eine representative Gastgeberstadt je Gruppe (nur fuer die Anzeige).
_GROUP_CITY = {
    "A": ("Mexico City", "Mexico"), "B": ("Toronto", "Canada"),
    "C": ("Los Angeles", "United States"), "D": ("Los Angeles", "United States"),
    "E": ("New York", "United States"), "F": ("San Francisco", "United States"),
    "G": ("Dallas", "United States"), "H": ("Miami", "United States"),
    "I": ("Atlanta", "United States"), "J": ("Houston", "United States"),
    "K": ("Vancouver", "Canada"), "L": ("Seattle", "United States"),
}

# Reihenfolge der drei Spieltage als Index-Paare innerhalb einer Gruppe.
# Teamindex 0 ist der gesetzte Pot-1-Kopf (bei Gastgebern automatisch heim).
_ROUND_ROBIN = [
    (0, 1), (2, 3),   # Spieltag 1
    (0, 2), (3, 1),   # Spieltag 2
    (0, 3), (1, 2),   # Spieltag 3
]


def all_teams() -> List[str]:
    """Alle 48 Teilnehmer als flache Liste (Reihenfolge = Gruppen A..L)."""
    return [t for g in GROUPS.values() for t in g]


def team_group() -> Dict[str, str]:
    """Mapping Team -> Gruppenbuchstabe."""
    return {t: g for g, teams in GROUPS.items() for t in teams}


def build_fixtures() -> pd.DataFrame:
    """Erzeugt den vollstaendigen Gruppenspielplan (72 Spiele).

    Pro Gruppe ein vollstaendiges Rundenturnier (6 Spiele). Spielt ein
    Gastgeber, wird er als Heimteam gefuehrt und das Spiel ist **nicht
    neutral** (Heimvorteil greift); sonst neutraler Platz.
    """
    rows = []
    match_id = 1
    for gi, (group, teams) in enumerate(GROUPS.items()):
        city, country = _GROUP_CITY[group]
        for slot, (i, j) in enumerate(_ROUND_ROBIN):
            home, away = teams[i], teams[j]
            # Gastgeber immer als Heimteam fuehren (Heimvorteil).
            if away in HOSTS and home not in HOSTS:
                home, away = away, home
            neutral = 0 if home in HOSTS else 1
            matchday = slot // 2  # 0,1,2
            fixture_date = GROUP_STAGE_START + timedelta(days=matchday * 5 + gi % 5)
            rows.append({
                "match_id": match_id,
                "date": fixture_date.strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "city": city,
                "country": country,
                "neutral": neutral,
                "group_name": group,
                "stage": "group",
            })
            match_id += 1
    return pd.DataFrame(rows)


def write_fixtures(path) -> "pd.DataFrame":
    """Schreibt den Spielplan als CSV und gibt ihn zurueck."""
    from pathlib import Path

    df = build_fixtures()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
