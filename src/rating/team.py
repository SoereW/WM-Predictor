"""Team-Aggregation: von Spielerscores zum Team-Gesamtscore.

Schritte:
1. Jeden Spieler bewerten (``rate_player``).
2. Beste Elf nach Zielformation zusammenstellen (``build_best_xi``).
3. Kadertiefe aus den naechststaerksten Spielern.
4. Chemie ueber das Chemie-Modul.
5. Trainer/Coaching-Score.
6. Gewichtete Kombination (``TeamWeights``) -> Team-Gesamtscore.

Der Team-Score ist bewusst auf derselben 0-100-Skala wie die Spieler, damit
"Team 84" intuitiv vergleichbar mit "Spieler 84" ist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .chemistry import ChemistryScore, compute_chemistry
from .criteria import (
    FORMATION,
    NEUTRAL_CHEMISTRY,
    ROLE_IMPORTANCE,
    STAR_ALPHA,
    ChemistryWeights,
    PlayerWeights,
    TeamWeights,
)
from .player import PlayerScore, rate_player


@dataclass
class TeamScore:
    team: str
    overall: float
    best_xi_score: float
    depth_score: float
    chemistry: ChemistryScore
    coach_score: float
    best_xi: List[PlayerScore] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)


def _coach_score(coach: Dict | None) -> float:
    """Coaching-Score 0-100 aus Reputation/Erfahrung (optional)."""
    if not coach:
        return 70.0
    if coach.get("rating") is not None:
        return float(np.clip(coach["rating"], 0, 100))
    tenure = float(coach.get("tenure_years", coach.get("tenure", 0)) or 0)
    majors = float(coach.get("major_tournaments", 0) or 0)
    base = 62.0 + 18.0 * (1.0 - np.exp(-tenure / 3.0)) + min(majors * 2.5, 15.0)
    return float(np.clip(base, 0, 100))


def best_xi_strength(best_xi: List[PlayerScore]) -> float:
    """Staerke der besten Elf - positions- und spitzengewichtet.

    Statt eines flachen Mittels werden (a) die Achsen-Positionen
    (TW/IV/Sechser/Stuermer) ueber ``ROLE_IMPORTANCE`` etwas hoeher gewichtet
    und (b) die staerksten Spieler ueber ``STAR_ALPHA`` zusaetzlich betont -
    so wie reale Mannschaftsstaerke entsteht. Beides ist mild gehalten.
    """
    if not best_xi:
        return 0.0
    ov = np.array([p.overall for p in best_xi], dtype=float)
    role_w = np.array([ROLE_IMPORTANCE.get(p.role, 1.0) for p in best_xi], dtype=float)
    role_w = role_w / role_w.sum()
    star_w = ov ** 2
    star_w = star_w / star_w.sum() if star_w.sum() > 0 else role_w
    w = (1.0 - STAR_ALPHA) * role_w + STAR_ALPHA * star_w
    w = w / w.sum()
    return float((ov * w).sum())


def build_best_xi(scored: List[PlayerScore], formation: Dict[str, int] | None = None) -> List[PlayerScore]:
    """Waehlt die beste Elf nach Formation; fuellt Rest mit den Besten."""
    formation = formation or FORMATION
    by_line: Dict[str, List[PlayerScore]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in scored:
        by_line.setdefault(p.position, by_line["MID"]).append(p) if p.position in by_line else by_line["MID"].append(p)
    for line in by_line:
        by_line[line].sort(key=lambda p: p.overall, reverse=True)

    chosen: List[PlayerScore] = []
    for line, n in formation.items():
        chosen.extend(by_line.get(line, [])[:n])
    # Auffuellen, falls eine Linie unterbesetzt ist.
    if len(chosen) < 11:
        chosen_ids = {id(p) for p in chosen}
        rest = sorted([p for p in scored if id(p) not in chosen_ids], key=lambda p: p.overall, reverse=True)
        chosen.extend(rest[: 11 - len(chosen)])
    return sorted(chosen, key=lambda p: p.overall, reverse=True)[:11]


def _xi_frame(squad: pd.DataFrame, best_xi: List[PlayerScore]) -> pd.DataFrame:
    """Findet die Kaderzeilen der besten Elf (fuer Chemie: club/caps/age)."""
    names = [p.name for p in best_xi]
    sub = squad[squad.get("player", squad.get("name")).isin(names)] if "player" in squad.columns or "name" in squad.columns else squad.head(11)
    return sub


def rate_team(
    squad: pd.DataFrame,
    coach: Dict | None = None,
    player_weights: PlayerWeights | None = None,
    team_weights: TeamWeights | None = None,
    chem_weights: ChemistryWeights | None = None,
    depth_n: int = 18,
) -> TeamScore:
    """Bewertet ein komplettes Team aus seinem Kader (+ optional Trainer)."""
    player_weights = player_weights or PlayerWeights()
    team_weights = team_weights or TeamWeights()

    team_name = str(squad["team"].iloc[0]) if "team" in squad.columns and len(squad) else ""
    scored = [rate_player(dict(r), player_weights) for _, r in squad.iterrows()]
    scored.sort(key=lambda p: p.overall, reverse=True)

    best_xi = build_best_xi(scored)
    best_xi_score = best_xi_strength(best_xi)

    depth = scored[:depth_n]
    depth_score = float(np.mean([p.overall for p in depth])) if depth else best_xi_score

    xi_frame = _xi_frame(squad, best_xi)
    chemistry = compute_chemistry(squad, xi_frame, coach=coach, weights=chem_weights)
    coach_sc = _coach_score(coach)

    # Niveau (reine Spielstaerke) aus Elf, Tiefe, Coaching.
    w = team_weights
    skill_weight = w.best_xi + w.depth + w.coach
    level = (w.best_xi * best_xi_score + w.depth * depth_score + w.coach * coach_sc) / skill_weight

    # Chemie wirkt als Modifikator *relativ zu neutraler Chemie* und ist an
    # das Niveau gekoppelt: gute Chemie hebt einen starken Kader spuerbar,
    # ueberdeckt aber bei einem schwachen Kader nicht die fehlende Qualitaet
    # (perfekte Chemie macht aus einem 73er-Kader keinen Top-Favoriten).
    chem_delta = (chemistry.overall - NEUTRAL_CHEMISTRY) / 100.0  # ~[-0.75, 0.25]
    chem_effect = w.chemistry * chem_delta * level  # in Score-Punkten
    overall = float(np.clip(level + chem_effect, 0, 100))

    return TeamScore(
        team=team_name,
        overall=round(overall, 1),
        best_xi_score=round(best_xi_score, 1),
        depth_score=round(depth_score, 1),
        chemistry=chemistry,
        coach_score=round(coach_sc, 1),
        best_xi=best_xi,
        components={
            "level": round(level, 1),
            "chemistry_effect": round(chem_effect, 2),
            "best_xi_part": round(w.best_xi * best_xi_score / skill_weight, 1),
            "depth_part": round(w.depth * depth_score / skill_weight, 1),
            "coach_part": round(w.coach * coach_sc / skill_weight, 1),
        },
    )


def rate_all_teams(
    squads: pd.DataFrame,
    coaches: Dict[str, Dict] | None = None,
    **kwargs,
) -> Dict[str, TeamScore]:
    """Bewertet alle Teams in einem Kader-DataFrame."""
    coaches = coaches or {}
    out: Dict[str, TeamScore] = {}
    for team, group in squads.groupby("team"):
        out[team] = rate_team(group.reset_index(drop=True), coach=coaches.get(team), **kwargs)
    return out
