"""Bewertungssystem fuer Spieler, Teams und Chemie.

Eigenstaendiges Subsystem, bewusst **getrennt** vom Vorhersagemodell
(``src/model.py``). Es beantwortet die Frage "wie stark ist dieser Kader
und wie gut passt er zusammen?" anhand klar definierter, gewichteter
Kriterien - unabhaengig davon, wie der Predictor Wahrscheinlichkeiten
erzeugt.

Bausteine:
- ``criteria``  : alle Kriterien, Positionsprofile und Gewichte (zentral)
- ``player``    : Spielerbewertung (Attribute -> Skill -> Gesamtscore)
- ``chemistry`` : Team-Chemie (Vereinsbloecke, Eingespieltheit, ...)
- ``team``      : Aggregation zur Teamstaerke inkl. Chemie und Trainer
- ``sample``    : kuratierte Naeherungsdaten bekannter Spieler (fuer Tests)
"""

from .criteria import (
    ATTRIBUTES,
    ChemistryWeights,
    PlayerWeights,
    TeamWeights,
    normalize_position,
    position_line,
)
from .player import PlayerScore, rate_player, rate_players
from .chemistry import ChemistryScore, compute_chemistry
from .team import TeamScore, build_best_xi, rate_team, rate_all_teams
from .io import load_squads_csv

__all__ = [
    "ATTRIBUTES",
    "PlayerWeights",
    "TeamWeights",
    "ChemistryWeights",
    "normalize_position",
    "position_line",
    "PlayerScore",
    "rate_player",
    "rate_players",
    "ChemistryScore",
    "compute_chemistry",
    "TeamScore",
    "build_best_xi",
    "rate_team",
    "rate_all_teams",
    "load_squads_csv",
]
