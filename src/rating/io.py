"""Laden echter Kaderdaten fuer das Bewertungssystem.

Erwartetes CSV-Format (Spalten case-insensitiv; fehlende optionale Spalten
werden mit sinnvollen Defaults gefuellt):

  Pflicht : team, player, position
  Skill   : pace, shooting, passing, dribbling, defending, physical, vision, gk
  Kontext : club, league, age, caps, form, fitness/available, natural_position

Fehlen einzelne Attribute, werden sie aus einem optionalen ``overall``-Wert
genaehert (gleichverteilt auf die Attribute), sonst neutral (60) gesetzt -
so laesst sich auch ein einfacher Datensatz mit nur ``overall`` nutzen.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .criteria import ATTRIBUTES, normalize_position, position_role
from ..teams import normalize_team


def load_squads_csv(csv_path) -> pd.DataFrame:
    """Liest eine Kader-CSV und vereinheitlicht Spalten/Typen."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "player" not in df.columns and "name" in df.columns:
        df["player"] = df["name"]
    required = {"team", "player", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kader-CSV fehlt Pflichtspalten: {sorted(missing)}")

    df["team"] = df["team"].map(normalize_team)
    df["name"] = df.get("player")
    df["natural_position"] = df.get("natural_position", df["position"])

    # Attribute auffuellen.
    has_overall = "overall" in df.columns
    for attr in ATTRIBUTES:
        if attr not in df.columns:
            if has_overall:
                df[attr] = pd.to_numeric(df["overall"], errors="coerce")
            else:
                df[attr] = 60.0
        df[attr] = pd.to_numeric(df[attr], errors="coerce").fillna(60.0)

    # Kontextspalten mit Defaults.
    defaults = {"club": "", "league": "", "age": 24, "caps": 0, "form": 70}
    for col, dv in defaults.items():
        if col not in df.columns:
            df[col] = dv
    if "fitness" not in df.columns and "available" not in df.columns:
        df["available"] = 1

    df = df.dropna(subset=["team", "player"])
    df = df[df["team"].astype(str) != ""]
    return df.reset_index(drop=True)
