"""Ingest echter Spielerdaten im FIFA/EA-Format -> Bewertungs-Schema.

Quelle: offen verfuegbarer FIFA-20-Spielerdatensatz (committet, kein
Login/LFS noetig). Enthaelt fuer ~18k Spieler die Attribute, die das
Bewertungssystem braucht (pace, shooting, passing, dribbling, defending,
physic) plus Verein, Liga, Nationalitaet, Alter und Overall.

WICHTIG / EHRLICHKEIT:
- Es sind **FIFA-20**-Werte (Saison 2019/20), nicht der aktuelle Stand.
  Sie dienen dazu, das Bewertungssystem mit *echten* Spielerattributen zu
  fuettern und die Logik zu pruefen - nicht als tagesaktuelle Wahrheit.
  Fuer die WM 2026 sollten aktuelle Kader/Ratings eingespeist werden.
- ``caps`` (Laenderspiele) sind im Datensatz nicht enthalten und werden aus
  ``international_reputation`` (1-5) genaehert; ``form`` wird aus dem Overall
  abgeleitet. Beides ist klar markiert und konservativ gewaehlt.

Das Format der Ausgabe passt exakt auf ``src/rating/io.load_squads_csv``
bzw. direkt auf ``rate_all_teams``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..teams import normalize_team

# Committeter FIFA-20-Datensatz (raw, ~9 MB, 18k Spieler, inkl. league_name).
FIFA20_URL = "https://raw.githubusercontent.com/305kishan/FIFA/main/data/FIFA20.csv"

# FIFA-Detailposition -> unser Positions-Kuerzel (criteria._POSITION_MAP kennt diese).
_POS_FIRST = {
    "GK": "GK",
    "CB": "CB", "RCB": "CB", "LCB": "CB",
    "RB": "RB", "LB": "LB", "RWB": "RB", "LWB": "LB",
    "CDM": "CDM", "RDM": "CDM", "LDM": "CDM",
    "CM": "CM", "RCM": "CM", "LCM": "CM",
    "CAM": "CAM", "RAM": "CAM", "LAM": "CAM",
    "RM": "RM", "LM": "LM",
    "RW": "RW", "LW": "LW",
    "RF": "RW", "LF": "LW", "CF": "CF",
    "ST": "ST", "RS": "ST", "LS": "ST",
}

# FIFA-Liganame -> Schluessel in criteria.LEAGUE_STRENGTH (sonst greift die
# liga-gewichtete Chemie nicht). Nur die wichtigsten; Rest faellt auf Default.
_LEAGUE_MAP = {
    "English Premier League": "Premier League",
    "Spain Primera Division": "La Liga",
    "Italian Serie A": "Serie A",
    "German 1. Bundesliga": "Bundesliga",
    "French Ligue 1": "Ligue 1",
    "Portuguese Liga ZON SAGRES": "Primeira Liga",
    "Holland Eredivisie": "Eredivisie",
    "Belgian Jupiler Pro League": "Belgian Pro League",
    "Turkish Süper Lig": "Super Lig",
    "English League Championship": "Championship",
    "USA Major League Soccer": "MLS",
    "Saudi Abdul L. Jameel League": "Saudi Pro League",
    "Mexican Liga MX": "Liga MX",
    "Japanese J. League Division 1": "J1 League",
    "Argentina Primera División": "Liga Profesional",
    "Brazilian Série A": "Brasileirao",
}


def _map_league(name: object) -> str:
    """FIFA-Liganame auf einen bekannten Liga-Schluessel abbilden."""
    s = str(name or "").strip()
    return _LEAGUE_MAP.get(s, s)


def fetch_fifa_players(dest, url: str = FIFA20_URL, retries: int = 4, timeout: int = 90) -> Path:
    """Laedt den FIFA-Spielerdatensatz herunter (mit Exponential-Backoff)."""
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return dest
        except Exception as err:  # noqa: BLE001 - bewusst breit, Netzwerk
            last_err = err
            wait = 2 ** (attempt + 1)
            print(f"Download fehlgeschlagen ({attempt + 1}/{retries}): {err} - warte {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Konnte FIFA-Spielerdaten nicht laden: {last_err}")


def _first_position(pos_field: object) -> str:
    """Erste (Haupt-)Position aus 'RW, CF, ST' -> 'RW'."""
    if pos_field is None or (isinstance(pos_field, float) and np.isnan(pos_field)):
        return "CM"
    first = str(pos_field).split(",")[0].strip().upper()
    return _POS_FIRST.get(first, "CM")


def _gk_rating(row: pd.Series) -> float:
    """Torwart-Skill aus den goalkeeping_*-Spalten (Mittel)."""
    cols = [
        "goalkeeping_diving", "goalkeeping_handling",
        "goalkeeping_positioning", "goalkeeping_reflexes",
    ]
    vals = [row[c] for c in cols if c in row.index and pd.notna(row[c])]
    if vals:
        return float(np.mean(vals))
    return float(row.get("overall", 70) or 70)


def load_fifa_as_squads(csv_path) -> pd.DataFrame:
    """Wandelt den FIFA-CSV in das Bewertungs-Schema (eine Zeile pro Spieler)."""
    df = pd.read_csv(csv_path, low_memory=False)

    name_col = "short_name" if "short_name" in df.columns else "long_name"
    nat_col = "nationality_name" if "nationality_name" in df.columns else "nationality"
    pos_col = "player_positions" if "player_positions" in df.columns else "team_position"
    club_col = "club_name" if "club_name" in df.columns else "club"

    out = pd.DataFrame()
    out["player"] = df[name_col].astype(str)
    out["name"] = out["player"]
    out["team"] = df[nat_col].map(normalize_team)
    out["position"] = df[pos_col].map(_first_position)
    out["natural_position"] = out["position"]
    out["club"] = df.get(club_col, "")
    out["league"] = df.get("league_name", pd.Series([""] * len(df))).map(_map_league)
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce").fillna(26)

    # Attribute (FIFA -> unser Schema).
    for src_col, dst in [
        ("pace", "pace"), ("shooting", "shooting"), ("passing", "passing"),
        ("dribbling", "dribbling"), ("defending", "defending"), ("physic", "physical"),
    ]:
        out[dst] = pd.to_numeric(df.get(src_col), errors="coerce")

    # vision: FIFA hat kein direktes Feld -> aus Passing/Overall genaehert.
    overall = pd.to_numeric(df.get("overall"), errors="coerce").fillna(65)
    out["vision"] = (0.6 * out["passing"].fillna(overall) + 0.4 * overall).clip(0, 99)
    out["gk"] = df.apply(_gk_rating, axis=1)

    # Feldspieler haben bei FIFA NaN-Felder fuer pace/shooting (nur GK) -> aus
    # Overall fuellen; GKs haben NaN bei pace/shooting -> niedrig ansetzen.
    is_gk = out["position"] == "GK"
    for col in ["pace", "shooting", "passing", "dribbling", "defending", "physical"]:
        out.loc[~is_gk, col] = out.loc[~is_gk, col].fillna(overall[~is_gk])
        out.loc[is_gk, col] = out.loc[is_gk, col].fillna(45.0)

    # caps (nicht im Datensatz) aus international_reputation 1-5 naehern.
    rep = pd.to_numeric(df.get("international_reputation"), errors="coerce").fillna(1)
    out["caps"] = ((rep - 1) * 22 + 8).clip(0, 120)  # rep1->8, rep5->96
    # form aus Overall (leichte Streuung um Mittel), konservativ.
    out["form"] = (overall.clip(40, 95)).astype(float)
    out["available"] = 1

    out = out.dropna(subset=["team", "player"])
    out = out[out["team"].astype(str) != ""]
    return out.reset_index(drop=True)
