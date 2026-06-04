"""Feature-Engineering ohne Information-Leakage.

Aus der Spielhistorie werden *Pre-Match*-Features gebaut. Neben Elo-Differenz
und einfacher Form (Punkte, Tore) auch **gegnerbezogene** Groessen:

- ``sos`` (strength of schedule): mittlere Gegnerstaerke der letzten Spiele -
  ein 2:0 gegen einen Topgegner ist mehr wert als gegen einen Aussenseiter.
- ``form_vs_exp``: erzielte minus *erwartete* Punkte (aus der Elo-Differenz).
  Misst, ob ein Team zuletzt ueber oder unter seinem Niveau gespielt hat -
  unabhaengig davon, wie stark die Gegner nominell waren.

Alle rollenden Groessen werden um ein Spiel verschoben, damit das aktuelle
Ergebnis nie in seine eigenen Features einfliesst. Die Gegnerstaerke nutzt
das *Pre-Match*-Elo (nur Vergangenheit) und ist damit ebenfalls leak-frei.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .elo import HOME_ADVANTAGE, EloModel, expected_score

ROLL_N = 10  # Anzahl letzter Spiele fuer Form-Features
HALF_LIFE_DAYS = 365.0 * 4.0  # Halbwertszeit fuer Zeitgewichtung (~4 Jahre)


def _result_points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def _team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Wandelt Spiele in eine Team-Perspektive (zwei Zeilen pro Spiel).

    Erwartet, dass ``matches`` bereits die Pre-Match-Elo-Spalten
    ``home_elo_pre``/``away_elo_pre`` enthaelt (fuer Gegnerstaerke und
    erwartete Punkte). Jede Team-Zeile bekommt eigenes/gegnerisches
    Pre-Elo sowie die aus der Elo-Differenz erwarteten Punkte.
    """
    hfa = np.where(matches["neutral"].values == 1, 0.0, HOME_ADVANTAGE)
    he, ae = matches["home_elo_pre"].values, matches["away_elo_pre"].values
    # Erwartete Punkte (0..3) aus dem Elo-Erwartungswert; Remis-Anteil grob
    # ueber eine feste Unentschieden-Wahrscheinlichkeit eingepreist.
    exp_home = np.array([expected_score(h + a_ - aw) for h, a_, aw in zip(he, hfa, ae)])
    p_draw = 0.26
    home_exp_pts = (exp_home - p_draw / 2.0).clip(0, 1) * 3.0
    away_exp_pts = ((1 - exp_home) - p_draw / 2.0).clip(0, 1) * 3.0

    home = pd.DataFrame(
        {
            "mid": matches["mid"].values, "date": matches["date"].values,
            "team": matches["home_team"].values, "gf": matches["home_score"].values,
            "ga": matches["away_score"].values, "is_home": (1 - matches["neutral"].values).clip(0, 1),
            "opp_elo": ae, "exp_pts": home_exp_pts,
        }
    )
    away = pd.DataFrame(
        {
            "mid": matches["mid"].values, "date": matches["date"].values,
            "team": matches["away_team"].values, "gf": matches["away_score"].values,
            "ga": matches["home_score"].values, "is_home": np.zeros(len(matches), dtype=int),
            "opp_elo": he, "exp_pts": away_exp_pts,
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = [_result_points(f, a) for f, a in zip(long["gf"], long["ga"])]
    long["pts_vs_exp"] = long["points"] - long["exp_pts"]
    return long


def _rolling_pre(long: pd.DataFrame) -> pd.DataFrame:
    """Rollende, um ein Spiel verschobene Form je Team."""
    long = long.sort_values(["team", "date", "mid"]).reset_index(drop=True)
    grp = long.groupby("team", group_keys=False)

    def roll(col: str) -> pd.Series:
        return grp[col].apply(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean())

    long["form_pts"] = roll("points")
    long["form_gf"] = roll("gf")
    long["form_ga"] = roll("ga")
    long["sos"] = roll("opp_elo")             # mittlere Gegnerstaerke (Spielplan-Haerte)
    long["form_vs_exp"] = roll("pts_vs_exp")  # Punkte ueber/unter Erwartung
    long["prev_date"] = grp["date"].shift(1)
    return long


def build_features(matches: pd.DataFrame, elo_model: EloModel | None = None):
    """Baut die Pre-Match-Feature-Tabelle (eine Zeile pro Spiel)."""
    df = matches.sort_values("date").reset_index(drop=True).copy()
    df["mid"] = np.arange(len(df))

    # Pre-Match-Elo ist rein sequenziell (nur Vergangenheit) und damit
    # leak-frei. Wenn kein Modell uebergeben wird - oder ein Modell, dessen
    # Pre-Match-Arrays nicht zu diesem (ggf. groesseren) Datensatz passen -
    # wird Elo frisch ueber alle Spiele gefittet. Das liefert fuer jedes
    # Spiel exakt die Ratings, die ein fortgeschriebenes Elo ergeben wuerde.
    if elo_model is None or len(elo_model.pre_home) != len(df):
        elo_model = EloModel().fit(df)
    df["home_elo_pre"] = elo_model.pre_home
    df["away_elo_pre"] = elo_model.pre_away

    long = _rolling_pre(_team_long(df))

    # Form je Team-Perspektive zurueck an die Spiele heften.
    keyed = long.set_index(["mid", "team"])
    feats: Dict[str, np.ndarray] = {}
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        idx = list(zip(df["mid"], df[team_col]))
        sub = keyed.reindex(idx)
        feats[f"{side}_form_pts"] = sub["form_pts"].values
        feats[f"{side}_form_gf"] = sub["form_gf"].values
        feats[f"{side}_form_ga"] = sub["form_ga"].values
        feats[f"{side}_sos"] = sub["sos"].values
        feats[f"{side}_form_vs_exp"] = sub["form_vs_exp"].values
        feats[f"{side}_prev_date"] = sub["prev_date"].values

    for k, v in feats.items():
        df[k] = v

    # Globale Mittel als Fallback fuer Teams ohne Historie.
    gf_mean = float(np.nanmean(df[["home_form_gf", "away_form_gf"]].values))
    ga_mean = float(np.nanmean(df[["home_form_ga", "away_form_ga"]].values))
    pts_mean = float(np.nanmean(df[["home_form_pts", "away_form_pts"]].values))
    sos_mean = float(np.nanmean(df[["home_sos", "away_sos"]].values))
    df = df.fillna(
        value={
            "home_form_gf": gf_mean, "away_form_gf": gf_mean,
            "home_form_ga": ga_mean, "away_form_ga": ga_mean,
            "home_form_pts": pts_mean, "away_form_pts": pts_mean,
            "home_sos": sos_mean, "away_sos": sos_mean,
            "home_form_vs_exp": 0.0, "away_form_vs_exp": 0.0,
        }
    )

    # Ruhetage (auf sinnvolle Spanne begrenzt).
    for side in ("home", "away"):
        prev = pd.to_datetime(df[f"{side}_prev_date"], errors="coerce")
        cur = pd.to_datetime(df["date"], errors="coerce")
        rest = (cur - prev).dt.days
        df[f"{side}_rest"] = rest.fillna(30).clip(0, 60)

    df["neutral"] = df["neutral"].fillna(0).astype(int)
    hfa = np.where(df["neutral"] == 1, 0.0, HOME_ADVANTAGE)
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"] + hfa
    df["form_pts_diff"] = df["home_form_pts"] - df["away_form_pts"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    # Gegnerbezogene Differenzen.
    df["sos_diff"] = (df["home_sos"] - df["away_sos"]) / 100.0   # auf Elo/100-Skala
    df["form_vs_exp_diff"] = df["home_form_vs_exp"] - df["away_form_vs_exp"]

    # Ziel: 0 = Auswaertssieg, 1 = Remis, 2 = Heimsieg.
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 2,
        np.where(df["home_score"] == df["away_score"], 1, 0),
    )

    # Zeitgewichtung: neuere Spiele zaehlen mehr.
    last = pd.to_datetime(df["date"]).max()
    age_days = (last - pd.to_datetime(df["date"])).dt.days.clip(lower=0)
    df["time_weight"] = np.power(0.5, age_days / HALF_LIFE_DAYS)

    return df, elo_model


CLASSIFIER_FEATURES = [
    "elo_diff", "form_pts_diff", "home_form_gf", "home_form_ga",
    "away_form_gf", "away_form_ga", "rest_diff", "neutral",
    "sos_diff", "form_vs_exp_diff",
]


def classifier_matrix(feat_df: pd.DataFrame) -> pd.DataFrame:
    return feat_df[CLASSIFIER_FEATURES].astype(float)


def stacked_goals_frame(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Pro Spiel zwei Zeilen (Heim-/Auswaertssicht) fuer das Poisson-Tor-Modell.

    Zielgroesse ist die Toranzahl der jeweiligen Mannschaft, erklaert durch
    Heimindikator, Elo-Vorteil, eigene Offensiv- und gegnerische
    Defensivform.
    """
    home = pd.DataFrame(
        {
            "is_home": (1 - feat_df["neutral"]).clip(0, 1).astype(float),
            "elo_adv": (feat_df["home_elo_pre"] - feat_df["away_elo_pre"]) / 100.0,
            "own_attack": feat_df["home_form_gf"].astype(float),
            "opp_defense": feat_df["away_form_ga"].astype(float),
            "own_form": feat_df["home_form_pts"].astype(float),
            "goals": feat_df["home_score"].astype(float),
            "weight": feat_df["time_weight"].astype(float),
        }
    )
    away = pd.DataFrame(
        {
            "is_home": np.zeros(len(feat_df), dtype=float),
            "elo_adv": (feat_df["away_elo_pre"] - feat_df["home_elo_pre"]) / 100.0,
            "own_attack": feat_df["away_form_gf"].astype(float),
            "opp_defense": feat_df["home_form_ga"].astype(float),
            "own_form": feat_df["away_form_pts"].astype(float),
            "goals": feat_df["away_score"].astype(float),
            "weight": feat_df["time_weight"].astype(float),
        }
    )
    return pd.concat([home, away], ignore_index=True)


GOALS_FEATURES = ["is_home", "elo_adv", "own_attack", "opp_defense", "own_form"]


@dataclass
class TeamState:
    elo: float
    form_gf: float
    form_ga: float
    form_pts: float


def team_state(matches: pd.DataFrame, elo_model: EloModel, team: str) -> TeamState:
    """Aktueller Zustand eines Teams (Elo + Form) aus der gesamten Historie."""
    sub_home = matches[matches["home_team"] == team]
    sub_away = matches[matches["away_team"] == team]
    rows = []
    for _, r in sub_home.iterrows():
        rows.append((r["date"], r["home_score"], r["away_score"]))
    for _, r in sub_away.iterrows():
        rows.append((r["date"], r["away_score"], r["home_score"]))
    rows.sort(key=lambda x: x[0])
    recent = rows[-ROLL_N:]
    if recent:
        gf = float(np.mean([g for _, g, _ in recent]))
        ga = float(np.mean([a for _, _, a in recent]))
        pts = float(np.mean([_result_points(g, a) for _, g, a in recent]))
    else:
        gf, ga, pts = 1.2, 1.2, 1.2
    return TeamState(elo=elo_model.current_rating(team), form_gf=gf, form_ga=ga, form_pts=pts)
