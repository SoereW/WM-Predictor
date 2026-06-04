"""Team-Chemie: wie gut passt der Kader zusammen?

Chemie ist ein eigenstaendiger Faktor neben dem reinen Koennen. Zwei gleich
starke Kader koennen sehr unterschiedlich funktionieren, je nachdem wie gut
sie eingespielt sind. Bewertet werden fuenf Teilkriterien (Gewichte in
``ChemistryWeights``), jeweils 0-100:

1. **club_blocks**   - Spieler aus denselben Vereinen bilden eingespielte
   Bloecke (z. B. mehrere Spieler von einem Top-Club). Misst die groesste
   Vereins-Konzentration in der besten Elf.
2. **cohesion**      - Eingespieltheit ueber gemeinsame Erfahrung
   (Laenderspiele/Caps der Stammelf): ein Kern mit vielen Caps hat
   eingespielte Automatismen.
3. **positional**    - spielen die Spieler auf ihrer Stammposition? Wer
   ausserhalb seiner Rolle spielt, kostet Chemie.
4. **age_balance**   - ausgewogene Altersstruktur (Mischung aus Routine und
   Frische, Peak ~24-30).
5. **coach_stability** - Kontinuitaet auf der Trainerbank (Amtszeit).

Rueckgabe ist ein ``ChemistryScore`` mit Gesamtwert und allen Teilwerten,
damit nachvollziehbar bleibt, woher die Chemie kommt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .criteria import AGE_PEAK_HIGH, AGE_PEAK_LOW, ChemistryWeights, league_strength


@dataclass
class ChemistryScore:
    overall: float
    parts: Dict[str, float] = field(default_factory=dict)
    weighted: Dict[str, float] = field(default_factory=dict)


def _club_blocks_score(xi: pd.DataFrame) -> float:
    """Groesste Vereins-Konzentration in der Elf -> Block-Chemie.

    Skaliert mit der Liga-Staerke der Bloecke: ein eingespielter Block aus
    einer Top-Liga ist mehr wert als aus einer schwaecheren. So bekommt ein
    Aussenseiter, dessen Kader komplett in einer schwaecheren Heimatliga
    spielt, nicht automatisch Top-Chemie.
    """
    if "club" not in xi.columns or xi["club"].isna().all():
        return 60.0  # neutraler Default ohne Vereinsangaben
    clubs = [str(c).strip() for c in xi["club"].dropna() if str(c).strip()]
    if not clubs:
        return 60.0
    counts = Counter(clubs)
    n = len(xi)
    leagues = xi.get("league")
    # Mittlere Liga-Staerke der Elf (0-1) als Qualitaetsfaktor der Bloecke.
    if leagues is not None:
        strengths = [league_strength(c, lg) for c, lg in zip(xi["club"], leagues)]
    else:
        strengths = [league_strength(c) for c in xi["club"]]
    league_factor = float(np.clip(np.mean(strengths) if strengths else 0.6, 0, 1))

    largest = max(counts.values())
    second = sorted(counts.values(), reverse=True)[1] if len(counts) > 1 else 0
    block_ratio = (largest + 0.5 * second) / n
    raw = 40.0 + block_ratio * 120.0
    # Liga-Qualitaet daempft schwache Bloecke (0.55 Liga -> ~0.8 Faktor).
    scaled = raw * (0.55 + 0.45 * league_factor)
    return float(np.clip(scaled, 0, 100))


def _cohesion_score(xi: pd.DataFrame) -> float:
    """Eingespieltheit aus den Caps der Stammelf (saettigend)."""
    if "caps" not in xi.columns:
        return 65.0
    caps = pd.to_numeric(xi["caps"], errors="coerce").fillna(0).to_numpy()
    avg = float(np.mean(caps)) if len(caps) else 0.0
    # ~40 Caps Schnitt -> solide eingespielt; >70 -> sehr eingespielt.
    return float(np.clip(100.0 * (1.0 - np.exp(-avg / 38.0)), 0, 100))


def _positional_score(xi: pd.DataFrame) -> float:
    """Anteil Spieler auf ihrer Stammposition (sonst Chemie-Abzug)."""
    if "on_position" in xi.columns:
        frac = pd.to_numeric(xi["on_position"], errors="coerce").fillna(1).clip(0, 1).mean()
        return float(np.clip(frac * 100.0, 0, 100))
    # Heuristik: stimmt die zugewiesene Linie mit der Stammlinie ueberein?
    if {"position", "natural_position"}.issubset(xi.columns):
        match = (xi["position"].astype(str).str[:1] == xi["natural_position"].astype(str).str[:1]).mean()
        return float(np.clip(match * 100.0, 0, 100))
    return 85.0  # ohne Angaben: meist spielen Spieler auf ihrer Position


def _age_balance_score(squad: pd.DataFrame) -> float:
    """Ausgewogenheit der Altersstruktur (Peak-Bereich, moderate Streuung)."""
    if "age" not in squad.columns:
        return 70.0
    ages = pd.to_numeric(squad["age"], errors="coerce").dropna().to_numpy()
    if len(ages) == 0:
        return 70.0
    mean_age = float(np.mean(ages))
    # Naehe zum Peak-Fenster.
    if AGE_PEAK_LOW <= mean_age <= AGE_PEAK_HIGH:
        center = 100.0
    else:
        dist = min(abs(mean_age - AGE_PEAK_LOW), abs(mean_age - AGE_PEAK_HIGH))
        center = float(np.clip(100.0 - dist * 7.0, 0, 100))
    # Eine gewisse Streuung (Routine + Talente) ist gut; zu homogen schwaecher.
    spread = float(np.std(ages))
    spread_bonus = float(np.clip((spread - 2.0) * 6.0, -10, 12))
    return float(np.clip(0.85 * center + spread_bonus, 0, 100))


def _coach_stability_score(coach: Dict | None) -> float:
    """Trainer-Kontinuitaet aus Amtszeit (Jahre) und optionaler Erfahrung."""
    if not coach:
        return 65.0
    tenure = float(coach.get("tenure_years", coach.get("tenure", 0)) or 0)
    # 0 J -> frisch (~45); 2-3 J -> eingespielt (~85); >4 J saettigt.
    base = 45.0 + 55.0 * (1.0 - np.exp(-tenure / 2.2))
    if coach.get("major_tournaments"):
        base += min(float(coach["major_tournaments"]) * 2.0, 8.0)
    return float(np.clip(base, 0, 100))


def compute_chemistry(
    squad: pd.DataFrame,
    best_xi: pd.DataFrame,
    coach: Dict | None = None,
    weights: ChemistryWeights | None = None,
) -> ChemistryScore:
    """Berechnet die Team-Chemie aus Kader, bester Elf und Trainer."""
    weights = weights or ChemistryWeights()
    parts = {
        "club_blocks": round(_club_blocks_score(best_xi), 1),
        "cohesion": round(_cohesion_score(best_xi), 1),
        "positional": round(_positional_score(best_xi), 1),
        "age_balance": round(_age_balance_score(squad), 1),
        "coach_stability": round(_coach_stability_score(coach), 1),
    }
    wd = weights.as_dict()
    weighted = {k: round(wd[k] * v, 2) for k, v in parts.items()}
    overall = float(np.clip(sum(weighted.values()), 0, 100))
    return ChemistryScore(overall=round(overall, 1), parts=parts, weighted=weighted)
