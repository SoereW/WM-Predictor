"""Stichproben-Auswertung des Bewertungssystems an bekannten Spielern/Teams.

Da sich Spieler-/Team-/Chemie-Bewertung nicht klassisch backtesten laesst
(es gibt keine "wahre" Zahl), pruefen wir die *Plausibilitaet* an einer
kuratierten Stichprobe bekannter Spieler:

1. Spieler-Rangliste je Position - sind die Topspieler oben?
2. Team-Rangliste - liegt die Reihenfolge im erwarteten Korridor?
3. Chemie-Aufschluesselung je Team.
4. Automatische Plausibilitaets-Checks (bestehen/fallen) mit Begruendung.

Aufruf:
    python scripts/evaluate_ratings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rating.player import rate_player
from src.rating.sample import COACHES, load_sample_players
from src.rating.team import rate_all_teams


def _line(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> None:
    players = load_sample_players()
    print(f"Stichprobe: {len(players)} bekannte Spieler aus {players['team'].nunique()} Teams")
    print("(Attribute sind Naeherungen ~ EA/SoFIFA-Niveau, kein offizieller Wert.)")

    # 1) Spieler-Topliste gesamt.
    scored = [rate_player(dict(r)) for _, r in players.iterrows()]
    scored.sort(key=lambda p: p.overall, reverse=True)
    _line("Top 12 Spieler (Gesamtscore)")
    for p in scored[:12]:
        print(f"  {p.overall:5.1f}  {p.name:24s} {p.team:13s} {p.role:3s} "
              f"(skill {p.skill:.0f}, form {p.form:.0f}, exp {p.experience:.0f})")

    # 2) Bester Spieler je Linie.
    _line("Bester Spieler je Position")
    for line in ("GK", "DEF", "MID", "FWD"):
        cands = [p for p in scored if p.position == line]
        if cands:
            b = cands[0]
            print(f"  {line}: {b.name} ({b.team}) {b.overall:.1f}")

    # 3) Team-Ranking inkl. Chemie.
    teams = rate_all_teams(players, coaches=COACHES)
    ranked = sorted(teams.values(), key=lambda t: t.overall, reverse=True)
    _line("Team-Ranking (Gesamtscore)")
    print(f"  {'Team':14s} {'Gesamt':>6s} {'XI':>5s} {'Tiefe':>5s} {'Chemie':>6s} {'Coach':>5s}")
    for t in ranked:
        print(f"  {t.team:14s} {t.overall:6.1f} {t.best_xi_score:5.1f} "
              f"{t.depth_score:5.1f} {t.chemistry.overall:6.1f} {t.coach_score:5.1f}")

    # 4) Chemie-Aufschluesselung (Beispiel: bestes und schwaechstes Team).
    _line("Chemie-Aufschluesselung (Teilkriterien)")
    for t in (ranked[0], ranked[-1]):
        print(f"  {t.team}: {t.chemistry.overall:.1f}")
        for k, v in t.chemistry.parts.items():
            print(f"      {k:16s} {v:5.1f}  (gewichtet {t.chemistry.weighted[k]:.1f})")

    # 5) Automatische Plausibilitaets-Checks.
    _line("Plausibilitaets-Checks")
    checks = []
    by_name = {p.name: p for p in scored}
    team_rank = {t.team: i for i, t in enumerate(ranked)}

    def check(desc, ok):
        checks.append(ok)
        print(f"  [{'OK ' if ok else 'XX '}] {desc}")

    # Topstuermer sollten klar ueber Aussenseiter-Stuermern liegen.
    if "Kylian Mbappe" in by_name and "Firas Al-Buraikan" in by_name:
        check("Mbappe > Al-Buraikan (Stuermer-Diskriminierung)",
              by_name["Kylian Mbappe"].overall > by_name["Firas Al-Buraikan"].overall + 8)
    # Eine Top-Nation sollte vor dem Aussenseiter rangieren.
    if "Argentina" in team_rank and "Saudi Arabia" in team_rank:
        check("Argentina vor Saudi Arabia im Team-Ranking",
              team_rank["Argentina"] < team_rank["Saudi Arabia"])
    if "France" in team_rank and "Saudi Arabia" in team_rank:
        check("France vor Saudi Arabia im Team-Ranking",
              team_rank["France"] < team_rank["Saudi Arabia"])
    # Frankreich-Chemie (Deschamps, langer Tenure) > Brasilien (frischer Coach).
    if "France" in teams and "Brazil" in teams:
        check("Frankreich coach_stability >= Brasilien (laengere Amtszeit)",
              teams["France"].chemistry.parts["coach_stability"]
              >= teams["Brazil"].chemistry.parts["coach_stability"])
    # Aussenseiter-Team sollte deutlich unter Top-Nation liegen.
    if "Argentina" in teams and "Saudi Arabia" in teams:
        check("Argentina-Score deutlich > Saudi-Arabien (>= 6 Punkte)",
              teams["Argentina"].overall >= teams["Saudi Arabia"].overall + 6)
    # Alle Scores muessen im gueltigen Bereich liegen.
    check("Alle Team-Scores in [0,100]", all(0 <= t.overall <= 100 for t in ranked))

    passed = sum(checks)
    print(f"\nErgebnis: {passed}/{len(checks)} Checks bestanden.")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
