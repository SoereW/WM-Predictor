"""Taktik-Profile und Stil-Matchups.

Zwei gleich starke Teams koennen sich unterschiedlich gut liegen - eine
Mannschaft "spielt einem Gegner in die Karten" oder eben nicht. Dieses
Modul modelliert das in zwei Schritten:

1. **Taktikprofil** je Team (0-100 je Dimension): Pressing-Intensitaet,
   Ballbesitz-Neigung, Tempo/Direktheit, Defensivblock-Stabilitaet,
   Fluegellastigkeit, Konterstaerke. Es kann aus den Spielerattributen
   abgeleitet (``profile_from_squad``) oder manuell gesetzt werden.

2. **Matchup**: bestimmte Stile kontern andere. Beispiele:
   - Hohes Pressing leidet gegen ballsicheres, schnelles Konterspiel.
   - Ein tiefer, stabiler Block bremst ballbesitzorientierte Gegner.
   - Fluegelspiel gegen eine fluegelschwache Abwehr ist im Vorteil.
   Die Regeln sind **gewichtet** und liefern eine kleine, gedeckelte
   Elo-Korrektur (Heimsicht).

Wie der Kontext ist Taktik ein Feinschliff und bewusst beschraenkt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .player import rate_player

# Obergrenze der Taktik-Korrektur (Elo-Punkte).
CAP_TACTICS = 35.0

DIMENSIONS = ["pressing", "possession", "tempo", "block", "width", "counter"]


@dataclass
class TacticProfile:
    pressing: float = 50.0      # Intensitaet des Anlaufens
    possession: float = 50.0    # Ballbesitz-/Aufbau-Neigung
    tempo: float = 50.0         # Direktheit/Vertikalitaet
    block: float = 50.0         # defensive Stabilitaet/Kompaktheit
    width: float = 50.0         # Fluegellastigkeit
    counter: float = 50.0       # Konter-/Umschaltstaerke

    def as_dict(self) -> Dict[str, float]:
        return {d: float(getattr(self, d)) for d in DIMENSIONS}


# Gewichtete Matchup-Regeln. Jede Regel: (dimA, dimB, gewicht).
# Interpretation: die *eigene* Staerke ``dimA`` interagiert mit der
# *gegnerischen* Eigenschaft ``dimB``. Der Beitrag ist proportional zum
# Produkt beider (jeweils relativ zum neutralen Mittel 50). Beispiel
# (counter, pressing, +0.9): ein konterstarkes Team profitiert *genau dann*,
# wenn der Gegner hoch presst - presst er gar nicht, gibt es keinen Bonus.
# Positives Gewicht = Vorteil, negatives = Nachteil fuer das Team mit dimA.
_RULES = [
    # Schnelles Konterspiel bestraft gegnerisches Hochpressing.
    ("counter", "pressing", 0.9),
    ("tempo", "pressing", 0.5),
    # Ballbesitz leidet an stabilem Block und an aggressivem Pressing.
    ("possession", "block", -0.7),
    ("possession", "pressing", -0.4),
    # Eigener stabiler Block bremst ballbesitzstarke Gegner.
    ("block", "possession", 0.6),
    # Fluegelspiel gegen fluegellastige (dort verwundbare) Abwehr.
    ("width", "width", 0.4),
    # Hohes Pressing gewinnt gegen aufbaustarke (ballspielende) Gegner, die
    # es unter Druck setzt - gegen tiefe Teams bringt Pressing wenig.
    ("pressing", "possession", 0.4),
]


def profile_from_squad(squad: pd.DataFrame) -> TacticProfile:
    """Leitet ein Taktikprofil aus den Spielerattributen der besten Elf ab.

    Heuristik auf Basis der aggregierten Attribute: viel Tempo/Dribbling ->
    Konter/Tempo; viel Passing/Vision -> Ballbesitz; viel Defending/Physis
    -> Block/Pressing; Fluegelstuermer -> width.
    """
    if squad is None or len(squad) == 0:
        return TacticProfile()
    scored = sorted([rate_player(dict(r)) for _, r in squad.iterrows()],
                    key=lambda p: p.overall, reverse=True)[:11]
    # Durchschnittsattribute der besten Elf aus den Rohwerten ziehen.
    names = {p.name for p in scored}
    xi = squad[squad.get("player", squad.get("name")).isin(names)]
    if xi.empty:
        xi = squad.head(11)

    def mean(col: str, default: float = 60.0) -> float:
        if col in xi.columns:
            v = pd.to_numeric(xi[col], errors="coerce").mean()
            return float(v) if v == v else default
        return default

    pace, drib = mean("pace"), mean("dribbling")
    passing, vision = mean("passing"), mean("vision")
    defending, physical = mean("defending"), mean("physical")

    # Anteil echter Fluegelspieler in der Elf.
    roles = [p.role for p in scored]
    width = 40.0 + 60.0 * (sum(r in ("W", "WM", "FB") for r in roles) / max(len(roles), 1))

    prof = TacticProfile(
        pressing=float(np.clip(0.6 * physical + 0.4 * defending, 0, 100)),
        possession=float(np.clip(0.6 * passing + 0.4 * vision, 0, 100)),
        tempo=float(np.clip(0.6 * pace + 0.4 * drib, 0, 100)),
        block=float(np.clip(0.6 * defending + 0.4 * physical, 0, 100)),
        width=float(np.clip(width, 0, 100)),
        counter=float(np.clip(0.5 * pace + 0.3 * drib + 0.2 * vision, 0, 100)),
    )
    return prof


def matchup_delta(home: TacticProfile, away: TacticProfile) -> float:
    """Taktik-Korrektur (Elo, Heimsicht) aus dem Stil-Matchup beider Teams.

    Pro Regel ``(dimA, dimB, w)``: Beitrag ~ w * (eigene dimA relativ zu 50)
    * (gegnerische dimB relativ zu 50). Die Interaktion ist also ein
    *Produkt* - ein Stil wirkt nur, wenn der passende Gegner-Stil vorhanden
    ist. Symmetrisch fuer den Gast (zieht ab), damit gleiche Profile 0 ergeben.
    """
    h, a = home.as_dict(), away.as_dict()

    def norm(v: float) -> float:
        return (v - 50.0) / 50.0  # -1..+1 um die neutrale Mitte

    score = 0.0
    for dim_a, dim_b, w in _RULES:
        score += w * norm(h[dim_a]) * norm(a[dim_b])   # Heim nutzt eigene Staerke
        score -= w * norm(a[dim_a]) * norm(h[dim_b])   # Gast ebenso (Gegenrichtung)
    elo = score / len(_RULES) * CAP_TACTICS * 4.0
    return float(max(-CAP_TACTICS, min(CAP_TACTICS, elo)))


@dataclass
class TacticResult:
    elo_delta: float
    home_profile: Dict[str, float] = field(default_factory=dict)
    away_profile: Dict[str, float] = field(default_factory=dict)


def compute_tactics(home_squad: pd.DataFrame, away_squad: pd.DataFrame) -> TacticResult:
    """Bequemer Wrapper: Profile aus Kadern ableiten und Matchup berechnen."""
    hp = profile_from_squad(home_squad)
    ap = profile_from_squad(away_squad)
    return TacticResult(
        elo_delta=round(matchup_delta(hp, ap), 1),
        home_profile={k: round(v, 1) for k, v in hp.as_dict().items()},
        away_profile={k: round(v, 1) for k, v in ap.as_dict().items()},
    )
