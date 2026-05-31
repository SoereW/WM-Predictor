"""Kaderstaerke: Spieler bewerten, zum Team aggregieren, mit Elo verrechnen.

Hintergrund: Elo misst nur, *wie die Nation historisch gespielt hat* - nicht,
*wie stark der Kader ist, der morgen auflaeuft*. Genau dort versagt reines
Elo: Kaderumbruch, Verletzungen, eine aufstrebende Generation oder lange
Laenderspielpausen. Dieses Modul holt dieses Signal herein.

Ablauf:
1. ``load_squads`` liest Spieler (team, player, position, rating, available).
2. ``team_overall_map`` aggregiert positionsbewusst zur Teamstaerke
   (beste Elf nach Formation + Kadertiefe).
3. ``calibrate_to_elo`` lernt eine lineare Abbildung Overall -> Elo-Skala
   auf den Teams, fuer die *sowohl* Elo *als auch* Kaderdaten vorliegen.
   Dadurch ist die Kaderstaerke automatisch in derselben Skala wie Elo -
   ohne willkuerliche Umrechnungskonstante.

Die eigentliche Verrechnung (moderate Korrektur des effektiven Ratings)
passiert im Modell (``src/model.py``) erst zum Vorhersagezeitpunkt. Die
trainierten ML-Modelle sehen die Kaderstaerke nie - sie kann daher nicht
overfitten, sondern nur den Elo-Input verschieben.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .teams import normalize_team

# Ziel-Formation fuer die "beste Elf". Robuste Standardaufstellung; sorgt
# dafuer, dass nicht z. B. elf Stuermer ausgewaehlt werden.
FORMATION = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}

# Anteil Stammelf vs. Kadertiefe an der Teamstaerke. Die Stammelf dominiert,
# aber etwas Tiefe (Auswechselspieler, Rotation, Verletzungspuffer) zaehlt.
XI_WEIGHT = 0.85
DEPTH_WEIGHT = 0.15
DEPTH_N = 18  # so viele Spieler fliessen in den Tiefe-Term ein

_POS_ALIASES = {
    "GK": "GK", "GOALKEEPER": "GK", "TW": "GK", "TOR": "GK",
    "DEF": "DEF", "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF",
    "RWB": "DEF", "DF": "DEF", "ABWEHR": "DEF", "VERTEIDIGER": "DEF",
    "MID": "MID", "CM": "MID", "CDM": "MID", "CAM": "MID", "LM": "MID",
    "RM": "MID", "MF": "MID", "MITTELFELD": "MID",
    "FWD": "FWD", "ST": "FWD", "CF": "FWD", "LW": "FWD", "RW": "FWD",
    "FW": "FWD", "STURM": "FWD", "ANGRIFF": "FWD",
}


def _pos_group(pos: object) -> str:
    """Bildet beliebige Positionsbezeichnungen auf GK/DEF/MID/FWD ab."""
    key = str(pos or "").strip().upper()
    return _POS_ALIASES.get(key, "MID")  # Unbekanntes konservativ ins Mittelfeld


def load_squads(csv_path) -> pd.DataFrame:
    """Liest eine Kader-CSV und vereinheitlicht Spalten/Typen.

    Erwartete Spalten (case-insensitiv): ``team``, ``player``, ``position``,
    ``rating``; optional ``available`` (1/0 - verletzte/gesperrte Spieler
    lassen sich so ausschliessen).
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"team", "player", "position", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kader-CSV fehlt Spalten: {sorted(missing)}")

    df["team"] = df["team"].map(normalize_team)
    df["position"] = df["position"].map(_pos_group)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "available" in df.columns:
        avail = df["available"].astype(str).str.lower().isin(["1", "true", "yes", "t", "ja"])
        df = df[avail]
    df = df.dropna(subset=["team", "player", "rating"])
    return df.reset_index(drop=True)


def _best_xi(group: pd.DataFrame) -> pd.DataFrame:
    """Waehlt die beste Elf nach Formation, fuellt Restplaetze mit den Besten."""
    chosen_idx: list = []
    for pos, n in FORMATION.items():
        pool = group[group["position"] == pos].sort_values("rating", ascending=False)
        chosen_idx.extend(pool.head(n).index.tolist())
    if len(chosen_idx) < 11:
        rest = group.drop(index=chosen_idx).sort_values("rating", ascending=False)
        chosen_idx.extend(rest.head(11 - len(chosen_idx)).index.tolist())
    return group.loc[chosen_idx].sort_values("rating", ascending=False).head(11)


def team_overall(group: pd.DataFrame) -> float:
    """Aggregierte Teamstaerke aus den Spielerratings einer Mannschaft."""
    if group.empty:
        return float("nan")
    xi = _best_xi(group)
    xi_mean = float(xi["rating"].mean())
    depth = group.sort_values("rating", ascending=False).head(DEPTH_N)
    depth_mean = float(depth["rating"].mean())
    return XI_WEIGHT * xi_mean + DEPTH_WEIGHT * depth_mean


def team_overall_map(squads: pd.DataFrame) -> Dict[str, float]:
    """Teamstaerke je Mannschaft als Dict."""
    return {team: team_overall(g) for team, g in squads.groupby("team")}


def calibrate_to_elo(
    overall_map: Dict[str, float], elo_ratings: Dict[str, float], min_teams: int = 5
) -> Tuple[float, float] | None:
    """Lernt Overall -> Elo (linear) auf Teams mit beiden Werten.

    Rueckgabe ``(a, b)`` fuer ``elo ~= a + b * overall`` oder ``None``, wenn
    zu wenige gemeinsame Teams vorhanden sind (dann wird keine
    Kaderkorrektur angewandt).
    """
    xs, ys = [], []
    for team, ovr in overall_map.items():
        if team in elo_ratings and ovr == ovr:  # ovr==ovr filtert NaN
            xs.append(ovr)
            ys.append(elo_ratings[team])
    if len(xs) < min_teams or np.std(xs) < 1e-6:
        return None
    b, a = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)
    return float(a), float(b)


def squad_to_elo(overall: float, calib: Tuple[float, float]) -> float:
    """Wandelt eine Teamstaerke in einen Elo-aequivalenten Wert um."""
    a, b = calib
    return a + b * overall
