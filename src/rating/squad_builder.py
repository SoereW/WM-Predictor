"""Nationalkader aus einem Spielerpool zusammenstellen.

Aus dem (echten) Spielerdatensatz wird je Nation der **staerkste verfuegbare
Kader** gebildet: positionsbewusst (genug Torhueter/Abwehr/Mittelfeld/
Angriff) und auf eine realistische Kadergroesse (Standard 26) begrenzt -
nach dem Vorbild einer echten WM-Nominierung.

EHRLICHKEIT: Das ist der *staerkste verfuegbare* Kader laut Spielerdaten,
nicht zwingend die offiziell nominierte 26er-Liste. Spieler aus weniger
abgedeckten Ligen fehlen in den Quelldaten - betroffene Nationen
(z. B. Iran, Jordanien, Usbekistan) haben daher duennere Kader; das wird in
der Auswertung als ``n_players`` transparent ausgewiesen. Die Auswahl ist
reproduzierbar und vollstaendig datengetrieben.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .criteria import normalize_position

# Positions-Quoten fuer einen 26er-Kader (Summe = 26).
DEFAULT_QUOTA: Dict[str, int] = {"GK": 3, "DEF": 9, "MID": 8, "FWD": 6}
DEFAULT_SIZE = 26

# Ab so vielen verfuegbaren Spielern gilt ein Kader als "voll abgedeckt"
# (Konfidenz 1.0). Darunter sinkt das Vertrauen linear - betroffen sind
# Nationen mit vielen Heimliga-Spielern, die im Datensatz fehlen.
COVERAGE_FULL = 18


def coverage_confidence(counts: Dict[str, int], full: int = COVERAGE_FULL) -> Dict[str, float]:
    """Datenabdeckung je Team -> Konfidenz 0..1 (linear, bei ``full`` = 1.0)."""
    return {t: min(1.0, n / float(full)) for t, n in counts.items()}


def squad_counts(squads) -> Dict[str, int]:
    """Anzahl Spieler je Team in einem Kader-DataFrame."""
    return {t: int(n) for t, n in squads.groupby("team").size().items()}


def _rank_col(df: pd.DataFrame) -> pd.Series:
    """Bestimmt die Spalte, nach der die Staerke sortiert wird."""
    for col in ("overall", "rating", "score"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # Fallback: Mittel der Feldattribute.
    attrs = [c for c in ("pace", "shooting", "passing", "dribbling", "defending", "physical") if c in df.columns]
    return df[attrs].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.0) if attrs else pd.Series(0.0, index=df.index)


def select_squad(
    team_df: pd.DataFrame,
    quota: Dict[str, int] | None = None,
    size: int = DEFAULT_SIZE,
) -> pd.DataFrame:
    """Waehlt den staerksten Kader einer Nation (positionsbewusst, gedeckelt).

    Vorgehen: je Linie (GK/DEF/MID/FWD) zunaechst die quotengemaesse Zahl der
    staerksten Spieler; danach Restplaetze bis ``size`` mit den naechstbesten
    Spielern auffuellen. Ist eine Linie unterbesetzt, wird genommen, was da
    ist (duennere Kader bleiben erlaubt und ehrlich).
    """
    quota = quota or DEFAULT_QUOTA
    if team_df.empty:
        return team_df

    df = team_df.copy()
    df["_line"] = df["position"].map(normalize_position)
    df["_rank"] = _rank_col(df)
    df = df.sort_values("_rank", ascending=False)

    chosen_idx: List = []
    for line, n in quota.items():
        pool = df[df["_line"] == line]
        chosen_idx.extend(pool.head(n).index.tolist())

    # Restplaetze bis zur Zielgroesse mit den besten uebrigen Spielern fuellen.
    if len(chosen_idx) < size:
        rest = df.drop(index=chosen_idx)
        chosen_idx.extend(rest.head(size - len(chosen_idx)).index.tolist())

    squad = df.loc[chosen_idx].sort_values("_rank", ascending=False)
    return squad.drop(columns=["_line", "_rank"]).reset_index(drop=True)


def build_squads(
    players: pd.DataFrame,
    teams: List[str],
    quota: Dict[str, int] | None = None,
    size: int = DEFAULT_SIZE,
) -> pd.DataFrame:
    """Baut die Kader fuer alle ``teams`` aus dem gemeinsamen Spielerpool."""
    parts = []
    for team in teams:
        team_df = players[players["team"] == team]
        if team_df.empty:
            continue
        parts.append(select_squad(team_df, quota=quota, size=size))
    if not parts:
        return players.iloc[0:0]
    return pd.concat(parts, ignore_index=True)
