"""Tests fuer das K.-o.-Bracket und die Drittel-Zuordnung."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import knockout as ko
from src.wm2026 import GROUPS


def test_topology_complete_and_consistent():
    """16 Sechzehntel-, 8 Achtel-, 4 Viertel-, 2 Halbfinale, 1 Finale."""
    assert len(ko.R32) == 16
    assert sum(1 for m in ko.ROUND_OF if ko.ROUND_OF[m] == "R16") == 8
    assert sum(1 for m in ko.ROUND_OF if ko.ROUND_OF[m] == "QF") == 4
    assert sum(1 for m in ko.ROUND_OF if ko.ROUND_OF[m] == "SF") == 2
    assert ko.ROUND_OF[ko.FINAL] == "Final"

    # Jedes spaetere Spiel speist sich aus zwei unterschiedlichen Kind-Spielen.
    for _mid, ((_, a), (_, b)) in ko.LATER.items():
        assert a != b

    # Vom Finale aus sind genau alle 31 K.-o.-Spiele erreichbar.
    seen: set = set()

    def walk(mid: int) -> None:
        seen.add(mid)
        if mid in ko.R32:
            return
        (_, a), (_, b) = ko.LATER[mid]
        walk(a)
        walk(b)

    walk(ko.FINAL)
    assert seen == ko.R32 | set(range(89, 103)) | {ko.FINAL}


def test_r32_slots_use_every_group_position_once():
    winners = [g for _m, (sh, sa) in ko.R32_SLOTS.items() for (k, g) in (sh, sa) if k == "W"]
    runners = [g for _m, (sh, sa) in ko.R32_SLOTS.items() for (k, g) in (sh, sa) if k == "R"]
    thirds = [1 for _m, (sh, sa) in ko.R32_SLOTS.items() for (k, _g) in (sh, sa) if k == "T"]
    assert sorted(winners) == sorted(GROUPS.keys())          # 12 Gruppensieger
    assert sorted(runners) == sorted(GROUPS.keys())          # 12 Gruppenzweite
    assert len(thirds) == 8                                   # 8 beste Dritte


def test_third_assignment_valid_for_all_495_combinations():
    """Fuer jede der C(12,8)=495 Drittel-Konstellationen entsteht ein gueltiges,
    regelkonformes Matching (kein Dritter trifft den eigenen Gruppensieger)."""
    groups = list(GROUPS.keys())
    combos = list(itertools.combinations(groups, 8))
    assert len(combos) == 495
    for combo in combos:
        assign = ko._assign_thirds(list(combo))
        assert len(assign) == 8
        assert set(assign.values()) == set(combo)            # alle 8 platziert, eindeutig
        for mid, g in assign.items():
            assert g != ko.THIRD_FORBIDDEN[mid]              # nie eigener Gruppensieger


def test_build_r32_yields_32_distinct_teams():
    winners = {g: f"{g}1" for g in GROUPS}
    runners = {g: f"{g}2" for g in GROUPS}
    combo = list(GROUPS.keys())[:8]
    third_by_group = {g: f"{g}3" for g in combo}
    assign = ko._assign_thirds(combo)
    pairs = ko._build_r32(winners, runners, third_by_group, assign)

    teams = [t for pr in pairs.values() for t in pr]
    assert len(teams) == 32
    assert len(set(teams)) == 32
