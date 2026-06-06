"""Tests fuer die K.-o.-Phase: Bracket-Struktur + Drittplatzierten-Zuordnung."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import knockout as ko  # noqa: E402

GROUPS = "ABCDEFGHIJKL"
_PASS = 0
_FAIL = 0


def check(name: str, cond: bool) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [OK ] {name}")
    else:
        _FAIL += 1
        print(f"  [XX ] {name}")


def test_r32_structure():
    """16 R32-Spiele, 32 Teilnehmer-Slots, korrekte Aufteilung."""
    check("test_r32_has_16_matches", len(ko.R32) == 16)
    slots = [s for pair in ko.R32.values() for s in pair]
    winners = [s for s in slots if s[0] == "W"]
    runners = [s for s in slots if s[0] == "R"]
    thirds = [s for s in slots if s[0] == "3"]
    check("test_12_group_winners", len(winners) == 12)
    check("test_12_runners_up", len(runners) == 12)
    check("test_8_thirds", len(thirds) == 8)
    # Jeder Gruppensieger und -zweite genau einmal.
    check("test_each_winner_once", sorted(w[1] for w in winners) == list(GROUPS))
    check("test_each_runner_once", sorted(r[1] for r in runners) == list(GROUPS))


def test_no_same_group_in_r32():
    """Ein Dritter trifft in R32 nie auf den Sieger seiner eigenen Gruppe."""
    ok = True
    for h, a in ko.R32.values():
        for slot, other in ((h, a), (a, h)):
            if slot[0] == "3" and other[0] == "W" and other[1] in slot[1]:
                ok = False
    check("test_third_eligible_excludes_own_group_winner", ok)


def test_bracket_tree_complete():
    """Spaetere Runden referenzieren ausschliesslich existierende Spiele."""
    ids = set(ko.R32) | set(ko.LATER)
    ok = all(src[1] in ids for pair in ko.LATER.values() for src in pair)
    check("test_later_rounds_reference_existing", ok)
    check("test_final_is_104", 104 in ko.LATER)
    check("test_round_of_covers_all", set(ko.ROUND_OF) == (set(range(73, 103)) | {104}))


def test_assign_thirds_all_combinations():
    """Fuer ALLE C(12,8)=495 Drittel-Kombinationen existiert ein gueltiges Matching."""
    n_total = 0
    n_valid = 0
    for combo in combinations(GROUPS, 8):
        n_total += 1
        assignment = ko.assign_thirds(list(combo))
        # Genau die acht Slots belegt.
        if set(assignment) != set(ko.THIRD_SLOTS):
            continue
        # Jede zugeordnete Gruppe ist im erlaubten Set des Slots.
        eligible = all(g in ko.THIRD_SLOTS[mid] for mid, g in assignment.items())
        # Bijektiv: jede der acht Gruppen genau einmal verwendet.
        bijective = sorted(assignment.values()) == sorted(combo)
        if eligible and bijective:
            n_valid += 1
    check("test_all_495_combinations_have_valid_matching", n_total == 495 and n_valid == 495)


def test_assign_thirds_deterministic():
    """Gleiche Eingabe -> gleiche Zuordnung (reproduzierbar)."""
    combo = list("ABCDEFGH")
    a1 = ko.assign_thirds(combo)
    a2 = ko.assign_thirds(combo)
    check("test_assignment_reproducible", a1 == a2)


def test_advance_prob():
    """Weiterkommens-Wahrscheinlichkeit ist konsistent und monoton."""
    # Symmetrisches Spiel: Remis fair aufgeteilt -> ~0.5.
    p = ko.advance_prob(0.4, 0.2, 0.4)
    check("test_advance_symmetric_half", abs(p - 0.5) < 1e-9)
    # Klarer Favorit kommt haeufiger weiter als der Aussenseiter.
    pf = ko.advance_prob(0.6, 0.25, 0.15)
    pu = ko.advance_prob(0.15, 0.25, 0.6)
    check("test_advance_complementary", abs((pf + pu) - 1.0) < 1e-9)
    check("test_favorite_advances_more", pf > 0.5 > pu)
    # Remis wird vollstaendig verteilt (Summe der Anteile = 1).
    check("test_no_probability_lost", abs(ko.advance_prob(0.5, 0.5, 0.0) - 1.0) < 1e-9)


def main():
    print("K.-o.-Phase Tests")
    test_r32_structure()
    test_no_same_group_in_r32()
    test_bracket_tree_complete()
    test_assign_thirds_all_combinations()
    test_assign_thirds_deterministic()
    test_advance_prob()
    print(f"\n{_PASS}/{_PASS + _FAIL} Tests bestanden.")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
