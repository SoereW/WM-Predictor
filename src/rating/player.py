"""Spielerbewertung: Attribute -> positionsspezifisches Skill -> Gesamtscore.

Der Gesamtscore eines Spielers entsteht in zwei Schritten:

1. **Skill**: Die Rohattribute (pace, shooting, ...) werden mit dem
   *Positionsprofil* der Rolle gewichtet (``ROLE_PROFILES``). Ein Stuermer
   wird so an Abschluss/Tempo gemessen, ein Innenverteidiger an
   Zweikampf/Physis. Ergebnis: ein positionsgerechtes Koennen 0-100.

2. **Gesamtscore**: Skill, aktuelle Form, Erfahrung und Fitness werden mit
   ``PlayerWeights`` kombiniert. So zaehlt nicht nur das nackte Talent,
   sondern auch Tagesform, Routine und Einsatzfaehigkeit.

Alle Teilwerte werden transparent zurueckgegeben (``PlayerScore``), damit
nachvollziehbar bleibt, *warum* ein Spieler welchen Score hat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .criteria import (
    ATTRIBUTES,
    ROLE_PROFILES,
    PlayerWeights,
    normalize_position,
    position_role,
)


@dataclass
class PlayerScore:
    name: str
    team: str
    position: str          # grobe Linie GK/DEF/MID/FWD
    role: str              # feines Rollenkuerzel
    skill: float           # positionsgewichtetes Koennen 0-100
    form: float            # aktuelle Form 0-100
    experience: float      # Erfahrung 0-100 (aus Caps + Alter)
    fitness: float         # Verfuegbarkeit/Fitness 0-100
    overall: float         # Gesamtscore 0-100
    components: Dict[str, float] = field(default_factory=dict)


def _experience_score(caps: float, age: float) -> float:
    """Erfahrung 0-100 aus Laenderspielen und Alter.

    Caps sind das Hauptsignal (Routine im Nationaltrikot); ein kleiner
    Altersanteil glaettet sehr junge Spieler mit wenigen Caps.
    """
    caps = float(caps or 0)
    age = float(age or 24)
    # 0 Caps -> 0 ; ~100 Caps -> ~90+ (saettigend).
    caps_part = 100.0 * (1.0 - np.exp(-caps / 45.0))
    # Alter: leichter Bonus Richtung Routine, gedeckelt.
    age_part = float(np.clip((age - 18) / (33 - 18), 0, 1) * 100.0)
    return float(np.clip(0.75 * caps_part + 0.25 * age_part, 0, 100))


def _skill_from_attributes(attrs: Dict[str, float], role: str) -> float:
    """Positionsgewichtetes Skill aus den Rohattributen."""
    profile = ROLE_PROFILES.get(role)
    if profile is None:
        # Fallback: Mittel ueber alle vorhandenen Attribute.
        vals = [float(attrs.get(a, 0) or 0) for a in ATTRIBUTES if a != "gk"]
        return float(np.clip(np.mean(vals) if vals else 0.0, 0, 100))
    score = 0.0
    for attr, w in profile.items():
        score += w * float(attrs.get(attr, 0) or 0)
    return float(np.clip(score, 0, 100))


def rate_player(row: Dict, weights: PlayerWeights | None = None) -> PlayerScore:
    """Bewertet einen einzelnen Spieler aus einem Dict/Series-Row."""
    weights = weights or PlayerWeights()
    name = str(row.get("player", row.get("name", "?")))
    team = str(row.get("team", ""))
    raw_pos = row.get("position", "MID")
    line = normalize_position(raw_pos)
    role = position_role(raw_pos)

    attrs = {a: float(row.get(a, 0) or 0) for a in ATTRIBUTES}
    skill = _skill_from_attributes(attrs, role)

    form = float(np.clip(row.get("form", 70) if row.get("form") is not None else 70, 0, 100))
    experience = _experience_score(row.get("caps", 0), row.get("age", 24))
    # Fitness: explizit oder aus 'available' abgeleitet.
    if row.get("fitness") is not None:
        fitness = float(np.clip(row.get("fitness"), 0, 100))
    else:
        avail = row.get("available", 1)
        fitness = 100.0 if str(avail).lower() in ("1", "true", "yes", "ja", "t") else 0.0

    w = weights
    overall = w.skill * skill + w.form * form + w.experience * experience + w.fitness * fitness
    overall = float(np.clip(overall, 0, 100))

    return PlayerScore(
        name=name, team=team, position=line, role=role,
        skill=round(skill, 1), form=round(form, 1),
        experience=round(experience, 1), fitness=round(fitness, 1),
        overall=round(overall, 1),
        components={
            "skill": round(w.skill * skill, 1),
            "form": round(w.form * form, 1),
            "experience": round(w.experience * experience, 1),
            "fitness": round(w.fitness * fitness, 1),
        },
    )


def rate_players(df: pd.DataFrame, weights: PlayerWeights | None = None) -> List[PlayerScore]:
    """Bewertet alle Spieler eines DataFrames."""
    weights = weights or PlayerWeights()
    return [rate_player(r._asdict() if hasattr(r, "_asdict") else dict(r), weights)
            for r in (row for _, row in df.iterrows())]
