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

# Koordinaten der Gastgeberstaedte (lat, lon) - fuer den Reiseweg-Kontext.
GROUP_VENUE_COORD = {
    "A": (19.43, -99.13), "B": (43.65, -79.38), "C": (34.05, -118.24),
    "D": (34.05, -118.24), "E": (40.71, -74.01), "F": (37.77, -122.42),
    "G": (32.78, -96.80), "H": (25.76, -80.19), "I": (33.75, -84.39),
    "J": (29.76, -95.37), "K": (49.28, -123.12), "L": (47.61, -122.33),
}

# Heimat-Koordinaten der 48 Nationen (repraesentativer Ort, lat/lon). Dienen
# der Reiseweg-Abschaetzung (Distanz Heimat -> Spielort): interkontinentale
# Teams reisen weiter, CONCACAF-Gastgeber sind lokal im Vorteil.
TEAM_COORD = {
    "Mexico": (19.43, -99.13), "South Africa": (-25.75, 28.19), "South Korea": (37.57, 126.98),
    "Czech Republic": (50.08, 14.44), "Canada": (45.42, -75.70), "Bosnia-Herzegovina": (43.86, 18.41),
    "Qatar": (25.29, 51.53), "Switzerland": (46.95, 7.45), "Brazil": (-15.79, -47.88),
    "Morocco": (34.02, -6.83), "Haiti": (18.59, -72.31), "Scotland": (55.95, -3.19),
    "United States": (38.90, -77.04), "Paraguay": (-25.30, -57.64), "Australia": (-35.28, 149.13),
    "Turkey": (39.93, 32.86), "Germany": (52.52, 13.40), "Curacao": (12.17, -68.99),
    "Ivory Coast": (6.83, -5.29), "Ecuador": (-0.18, -78.47), "Netherlands": (52.37, 4.90),
    "Japan": (35.68, 139.69), "Sweden": (59.33, 18.07), "Tunisia": (36.81, 10.18),
    "Belgium": (50.85, 4.35), "Egypt": (30.04, 31.24), "Iran": (35.69, 51.39),
    "New Zealand": (-41.29, 174.78), "Spain": (40.42, -3.70), "Cape Verde": (14.93, -23.51),
    "Saudi Arabia": (24.71, 46.68), "Uruguay": (-34.90, -56.16), "France": (48.85, 2.35),
    "Senegal": (14.72, -17.47), "Norway": (59.91, 10.75), "Iraq": (33.31, 44.36),
    "Argentina": (-34.60, -58.38), "Algeria": (36.75, 3.06), "Austria": (48.21, 16.37),
    "Jordan": (31.95, 35.93), "Portugal": (38.72, -9.14), "DR Congo": (-4.32, 15.31),
    "Uzbekistan": (41.30, 69.24), "Colombia": (4.71, -74.07), "England": (51.51, -0.13),
    "Croatia": (45.81, 15.98), "Ghana": (5.60, -0.19), "Panama": (8.98, -79.52),
}


def venue_coord(group: str):
    """Spielort-Koordinaten (lat, lon) einer Gruppe."""
    return GROUP_VENUE_COORD.get(group)

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
