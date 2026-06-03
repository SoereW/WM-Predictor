"""Tests fuer die WM-2026-Definition und den Kaderbau (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rating.criteria import normalize_position
from src.rating.squad_builder import build_squads, select_squad
from src.teams import normalize_team
from src.wm2026 import GROUPS, HOSTS, all_teams, build_fixtures, team_group

_RESULTS = []


def _check(name: str, cond: bool) -> None:
    _RESULTS.append((name, bool(cond)))
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}")


def test_groups_structure() -> None:
    _check("test_groups_12", len(GROUPS) == 12)
    teams = all_teams()
    _check("test_teams_48", len(teams) == 48)
    _check("test_teams_unique", len(set(teams)) == 48)
    _check("test_four_per_group", all(len(v) == 4 for v in GROUPS.values()))
    # Teamnamen sind bereits kanonisch (normalize ist idempotent).
    _check("test_names_canonical", all(normalize_team(t) == t for t in teams))


def test_fixtures() -> None:
    fx = build_fixtures()
    _check("test_72_fixtures", len(fx) == 72)
    _check("test_6_per_group", (fx.groupby("group_name").size() == 6).all())
    _check("test_no_self_play", (fx["home_team"] != fx["away_team"]).all())
    # Gastgeber-Heimspiele sind nicht neutral; reine Nicht-Gastgeber-Spiele schon.
    host_home = fx[fx["home_team"].isin(HOSTS)]
    _check("test_host_home_not_neutral", (host_home["neutral"] == 0).all())
    non_host = fx[~fx["home_team"].isin(HOSTS) & ~fx["away_team"].isin(HOSTS)]
    _check("test_non_host_neutral", (non_host["neutral"] == 1).all())
    # Jede Gruppe deckt genau ihre vier Teams ab.
    tg = team_group()
    ok = True
    for g, teams in GROUPS.items():
        seen = set(fx[fx["group_name"] == g]["home_team"]) | set(fx[fx["group_name"] == g]["away_team"])
        ok = ok and seen == set(teams)
    _check("test_group_team_coverage", ok)


def _toy_pool() -> pd.DataFrame:
    rows = []
    for i in range(30):
        pos = ["GK", "CB", "LB", "CM", "CAM", "ST", "RW"][i % 7]
        rows.append({"team": "Testland", "player": f"P{i}", "position": pos,
                     "overall": 80 - i, "pace": 70, "shooting": 70, "passing": 70,
                     "dribbling": 70, "defending": 70, "physical": 70, "vision": 70, "gk": 70})
    return pd.DataFrame(rows)


def test_squad_builder() -> None:
    squad = select_squad(_toy_pool(), size=26)
    _check("test_squad_capped", len(squad) <= 26)
    lines = squad["position"].map(normalize_position)
    _check("test_squad_has_gk", (lines == "GK").any())
    _check("test_squad_has_def", (lines == "DEF").any())
    # Staerkste Spieler werden bevorzugt (hoechster Overall ist dabei).
    _check("test_squad_keeps_best", squad["overall"].max() == 80)
    # Mehrere Teams bauen.
    pool = _toy_pool()
    pool2 = pool.assign(team="Andere")
    both = build_squads(pd.concat([pool, pool2], ignore_index=True), ["Testland", "Andere"])
    _check("test_build_two_teams", set(both["team"]) == {"Testland", "Andere"})


def main() -> None:
    print("WM-2026-Tests")
    test_groups_structure()
    test_fixtures()
    test_squad_builder()
    passed = sum(1 for _, ok in _RESULTS if ok)
    print(f"\n{passed}/{len(_RESULTS)} Tests bestanden.")
    if passed != len(_RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
