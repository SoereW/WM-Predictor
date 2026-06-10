"""K.-o.-Phase der WM 2026: Bracket, Turnier-Simulation und bester Turnierbaum.

Das WM-Format 2026: 12 Gruppen a 4 Teams; weiter kommen die zwei Besten je
Gruppe (24 Teams) plus die acht besten Gruppendritten -> 32 Teams in einer
reinen K.-o.-Runde:

    Sechzehntelfinale (R32, 16 Spiele 73-88)
    -> Achtelfinale     (R16, 8 Spiele 89-96)
    -> Viertelfinale    (QF,  4 Spiele 97-100)
    -> Halbfinale       (SF,  2 Spiele 101-102)
    -> Finale           (Spiel 104; Spiel 103 = Spiel um Platz 3)

Die Bracket-Topologie (welche Slots in welchem Spiel antreten, wie die Sieger
weiterwandern) folgt dem offiziellen FIFA-Spielplan-Format: 8 Spiele
*Gruppensieger gegen Gruppendritten*, 4 Spiele *Sieger gegen Zweiten* und
4 Spiele *Zweiter gegen Zweiten*; die zwoelf Gruppensieger sind ausgewogen
auf beide Turnierhaelften verteilt. Die Zuordnung der acht besten Dritten zu
den acht "Sieger-gegen-Dritter"-Slots erfolgt **regelkonform** (ein Dritter
trifft nie auf den Sieger seiner eigenen Gruppe) ueber ein Matching - genau
wie die FIFA-Kombinationstabelle (Annex), da erst nach der Gruppenphase
feststeht, aus welchen Gruppen die Dritten kommen.

> Hinweis: Die exakte Slot-Buchstaben-Zuordnung der FIFA-Annex-Tabelle laesst
> sich offline nicht final verifizieren; ``R32_SLOTS`` ist der **eine** Ort,
> um sie bei Bedarf gegen die offizielle Quelle abzugleichen. Topologie,
> Simulation und Turnierbaum sind davon unabhaengig korrekt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .teams import normalize_team

# ===========================================================================
# Bracket-Topologie.
# ===========================================================================
# Slot-Typen: ("W", group) = Gruppensieger, ("R", group) = Gruppenzweiter,
# ("T", None) = einer der acht besten Dritten (wird zugeordnet).
R32_SLOTS: Dict[int, Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str]]]] = {
    73: (("W", "A"), ("T", None)),
    74: (("R", "B"), ("R", "C")),
    75: (("W", "D"), ("T", None)),
    76: (("W", "E"), ("R", "F")),
    77: (("W", "G"), ("T", None)),
    78: (("R", "H"), ("R", "I")),
    79: (("W", "J"), ("T", None)),
    80: (("W", "K"), ("R", "L")),
    81: (("W", "B"), ("T", None)),
    82: (("W", "C"), ("R", "A")),
    83: (("W", "F"), ("T", None)),
    84: (("R", "D"), ("R", "E")),
    85: (("W", "H"), ("T", None)),
    86: (("W", "I"), ("R", "G")),
    87: (("W", "L"), ("T", None)),
    88: (("R", "J"), ("R", "K")),
}

# Spaeter Runden: Spiel -> ((Label, Kind-Spiel-Heim), (Label, Kind-Spiel-Gast)).
LATER: Dict[int, Tuple[Tuple[str, int], Tuple[str, int]]] = {
    89: (("Sieger 73", 73), ("Sieger 74", 74)),
    90: (("Sieger 75", 75), ("Sieger 76", 76)),
    91: (("Sieger 77", 77), ("Sieger 78", 78)),
    92: (("Sieger 79", 79), ("Sieger 80", 80)),
    93: (("Sieger 81", 81), ("Sieger 82", 82)),
    94: (("Sieger 83", 83), ("Sieger 84", 84)),
    95: (("Sieger 85", 85), ("Sieger 86", 86)),
    96: (("Sieger 87", 87), ("Sieger 88", 88)),
    97: (("Sieger 89", 89), ("Sieger 90", 90)),
    98: (("Sieger 91", 91), ("Sieger 92", 92)),
    99: (("Sieger 93", 93), ("Sieger 94", 94)),
    100: (("Sieger 95", 95), ("Sieger 96", 96)),
    101: (("Sieger 97", 97), ("Sieger 98", 98)),
    102: (("Sieger 99", 99), ("Sieger 100", 100)),
    104: (("Sieger 101", 101), ("Sieger 102", 102)),
}

R32 = set(R32_SLOTS.keys())
FINAL = 104

ROUND_ORDER = ["R32", "R16", "QF", "SF", "Final"]
ROUND_LABEL = {
    "R32": "Sechzehntelfinale",
    "R16": "Achtelfinale",
    "QF": "Viertelfinale",
    "SF": "Halbfinale",
    "Final": "Finale",
}


def _round_of() -> Dict[int, str]:
    out = {mid: "R32" for mid in R32}
    out.update({mid: "R16" for mid in range(89, 97)})
    out.update({mid: "QF" for mid in range(97, 101)})
    out.update({mid: "SF" for mid in (101, 102)})
    out[FINAL] = "Final"
    return out


ROUND_OF = _round_of()

# Repraesentatives Datum je K.-o.-Spiel (nur fuer die Anzeige).
_KO_DATE = {}
for _i, _mid in enumerate(sorted(R32)):
    _KO_DATE[_mid] = f"2026-06-{28 + _i // 3:02d}"  # 28.06.-03.07.
for _i, _mid in enumerate(range(89, 97)):
    _KO_DATE[_mid] = f"2026-07-{4 + _i // 2:02d}"   # 04.07.-07.07.
for _i, _mid in enumerate(range(97, 101)):
    _KO_DATE[_mid] = f"2026-07-{9 + _i // 2:02d}"   # 09.07.-10.07.
_KO_DATE[101], _KO_DATE[102] = "2026-07-14", "2026-07-15"
_KO_DATE[FINAL] = "2026-07-19"

# Dritten-Slots (die acht "Sieger gegen Dritter"-Spiele) und die jeweils
# verbotene Gruppe (der Gegner-Gruppensieger - kein Dritter trifft die eigene
# Gruppe).
THIRD_SLOTS = [mid for mid, (sh, sa) in R32_SLOTS.items() if sa[0] == "T" or sh[0] == "T"]
THIRD_FORBIDDEN: Dict[int, str] = {}
for _mid, (_sh, _sa) in R32_SLOTS.items():
    if _sh[0] == "T":
        THIRD_FORBIDDEN[_mid] = _sa[1]
    elif _sa[0] == "T":
        THIRD_FORBIDDEN[_mid] = _sh[1]


# ===========================================================================
# Vorbereitung: Team-Zustaende einmalig berechnen.
# ===========================================================================
@dataclass
class Prepared:
    """Einmalig vorberechnete Zustaende fuer schnelle Turnier-Durchlaeufe."""

    states: Dict = field(default_factory=dict)
    teams: List[str] = field(default_factory=list)


def prepare_tournament(predictor, history: pd.DataFrame, groups: Dict[str, List[str]]) -> Prepared:
    """Berechnet je Team einmalig seinen ``TeamState`` (Elo + Form).

    Danach laesst sich jedes beliebige Duell ohne erneutes Durchsuchen der
    Historie vorhersagen (``predictor.matchup_from_states``).
    """
    from .features import team_state

    teams = [normalize_team(t) for g in groups.values() for t in g]
    states = {t: team_state(history, predictor.elo, t) for t in teams}
    return Prepared(states=states, teams=teams)


def _advance_prob(p_home: float, p_draw: float, p_away: float) -> float:
    """Weiterkommens-Wahrscheinlichkeit des Heimteams im K.-o. (kein Remis).

    Der Remis-Anteil wird ueber die relative Staerke (Verlaengerung/Elfmeter)
    auf beide Teams aufgeteilt.
    """
    denom = p_home + p_away
    if denom <= 1e-9:
        return p_home + 0.5 * p_draw
    return p_home + p_draw * (p_home / denom)


# ===========================================================================
# Gruppenstruktur aus dem Spielplan.
# ===========================================================================
def _group_games(fixtures: pd.DataFrame, groups: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str, int]]]:
    """Pro Gruppe die Liste ihrer Spiele ``(home, away, neutral)``."""
    out: Dict[str, List[Tuple[str, str, int]]] = {g: [] for g in groups}
    if fixtures is None or len(fixtures) == 0:
        # Fallback: vollstaendiges Rundenturnier, neutraler Platz.
        for g, teams in groups.items():
            tt = [normalize_team(t) for t in teams]
            for i in range(len(tt)):
                for j in range(i + 1, len(tt)):
                    out[g].append((tt[i], tt[j], 1))
        return out
    for _, fx in fixtures.iterrows():
        g = fx.get("group_name")
        if g in out:
            out[g].append((normalize_team(fx["home_team"]), normalize_team(fx["away_team"]), int(fx.get("neutral", 1) or 0)))
    return out


def _expected_standings(predictor, group_games, prepared) -> Dict[str, List[Tuple[str, float, float, float]]]:
    """Erwartete Gruppentabelle je Gruppe (Punkte/Tordiff/Tore), absteigend."""
    out = {}
    for g, games in group_games.items():
        pts: Dict[str, float] = {}
        gd: Dict[str, float] = {}
        gf: Dict[str, float] = {}
        for home, away, neutral in games:
            ph, pdr, pa, lh, la = predictor.matchup_from_states(home, away, neutral, prepared.states)
            for t in (home, away):
                pts.setdefault(t, 0.0); gd.setdefault(t, 0.0); gf.setdefault(t, 0.0)
            pts[home] += 3 * ph + pdr
            pts[away] += 3 * pa + pdr
            gd[home] += lh - la
            gd[away] += la - lh
            gf[home] += lh
            gf[away] += la
        ranking = sorted(pts.keys(), key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
        out[g] = [(t, pts[t], gd[t], gf[t]) for t in ranking]
    return out


def _assign_thirds(third_groups: List[str]) -> Dict[int, str]:
    """Ordnet die acht besten Dritten (per Gruppenbuchstabe, beste zuerst) den
    acht Dritten-Slots zu - regelkonform (Dritter != Gegner-Gruppensieger).

    Rueckgabe: ``{match_id: gruppenbuchstabe}``.
    """
    slots = THIRD_SLOTS
    assignment: Dict[int, str] = {}
    used: set = set()

    def backtrack(i: int) -> bool:
        if i == len(slots):
            return True
        sid = slots[i]
        forbidden = THIRD_FORBIDDEN.get(sid)
        for g in third_groups:
            if g in used or g == forbidden:
                continue
            used.add(g)
            assignment[sid] = g
            if backtrack(i + 1):
                return True
            used.discard(g)
            del assignment[sid]
        return False

    backtrack(0)
    return assignment


def _build_r32(winners: Dict[str, str], runners: Dict[str, str],
               third_by_group: Dict[str, str], third_assign: Dict[int, str]) -> Dict[int, Tuple[str, str]]:
    """Fuellt die 16 Sechzehntelfinal-Paarungen aus den Gruppenplatzierungen."""
    def resolve(slot, mid):
        kind, grp = slot
        if kind == "W":
            return winners.get(grp, "")
        if kind == "R":
            return runners.get(grp, "")
        return third_by_group.get(third_assign.get(mid, ""), "")

    return {mid: (resolve(sh, mid), resolve(sa, mid)) for mid, (sh, sa) in R32_SLOTS.items()}


def _best_eight_thirds(thirds: List[Tuple[str, str, float, float, float]]) -> List[Tuple[str, str, float, float, float]]:
    """Sortiert die zwoelf Dritten (team, group, pts, gd, gf) und nimmt die acht besten."""
    return sorted(thirds, key=lambda r: (r[2], r[3], r[4]), reverse=True)[:8]


# ===========================================================================
# Wahrscheinlichster Turnierbaum (deterministisch aus Erwartungswerten).
# ===========================================================================
def most_likely_bracket(predictor, history, groups, fixtures=None, prepared=None):
    """Deterministischer Turnierbaum: je Spiel kommt der Favorit weiter.

    Rueckgabe: ``(rows, champion)`` mit einer Zeile je K.-o.-Spiel.
    """
    prepared = prepared or prepare_tournament(predictor, history, groups)
    group_games = _group_games(fixtures, groups)
    standings = _expected_standings(predictor, group_games, prepared)

    winners = {g: standings[g][0][0] for g in standings}
    runners = {g: standings[g][1][0] for g in standings}
    thirds_all = [(standings[g][2][0], g, standings[g][2][1], standings[g][2][2], standings[g][2][3]) for g in standings]
    best_thirds = _best_eight_thirds(thirds_all)
    third_by_group = {g: team for (team, g, *_rest) in best_thirds}
    third_assign = _assign_thirds([g for (_t, g, *_r) in best_thirds])

    pairs = _build_r32(winners, runners, third_by_group, third_assign)

    result_winner: Dict[int, str] = {}
    rows: List[Dict] = []

    def teams_of(mid: int) -> Tuple[str, str]:
        if mid in R32:
            return pairs[mid]
        (_, ch), (_, ca) = LATER[mid]
        return result_winner[ch], result_winner[ca]

    for mid in sorted(R32) + [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, FINAL]:
        home, away = teams_of(mid)
        # K.-o.-Spiele finden auf neutralem Platz statt (kein Heimvorteil).
        ph, pdr, pa, _lh, _la = predictor.matchup_from_states(home, away, 1, prepared.states)
        p_adv = _advance_prob(ph, pdr, pa)
        winner = home if p_adv >= 0.5 else away
        result_winner[mid] = winner
        rnd = ROUND_OF[mid]
        rows.append({
            "match_id": mid, "round": rnd, "round_label": ROUND_LABEL[rnd],
            "date": _KO_DATE[mid], "home": home, "away": away,
            "p_home_advance": round(float(p_adv), 3), "winner": winner,
        })

    return rows, result_winner[FINAL]


def bracket_to_frame(bracket) -> pd.DataFrame:
    """Bringt den Turnierbaum (Liste von Zeilen oder DataFrame) auf ein Schema."""
    cols = ["match_id", "round", "round_label", "date", "home", "away", "p_home_advance", "winner"]
    df = bracket if isinstance(bracket, pd.DataFrame) else pd.DataFrame(bracket)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].sort_values("match_id").reset_index(drop=True)


# ===========================================================================
# Monte-Carlo-Turnier-Simulation.
# ===========================================================================
@dataclass
class TournamentResult:
    probs: pd.DataFrame
    groups: pd.DataFrame


def _prep_group_grids(predictor, group_games, prepared):
    """Pro Gruppenspiel das kumulierte Tor-Gitter zum schnellen Sampling."""
    grids = {}
    for g, games in group_games.items():
        prepared_games = []
        for home, away, neutral in games:
            m = predictor.score_grid_from_states(home, away, neutral, prepared.states)
            kp1 = m.shape[0]
            prepared_games.append((home, away, np.cumsum(m.flatten()), kp1))
        grids[g] = prepared_games
    return grids


def simulate_tournament(predictor, history, groups, fixtures=None, n: int = 2000, prepared=None) -> TournamentResult:
    """Simuliert Gruppen **und** K.-o.-Phase in einem Durchlauf (Monte Carlo).

    Liefert je Nation P(Sechzehntel…Finale, **Titel**) sowie die
    Weiterkommens-Wahrscheinlichkeiten je Gruppe.
    """
    prepared = prepared or prepare_tournament(predictor, history, groups)
    group_games = _group_games(fixtures, groups)
    grids = _prep_group_grids(predictor, group_games, prepared)
    rng = np.random.default_rng(2026)

    group_of = {normalize_team(t): g for g, teams in groups.items() for t in teams}
    all_teams = list(group_of.keys())

    # Zaehler.
    reach = {r: {t: 0 for t in all_teams} for r in ("R32", "R16", "QF", "SF", "Final", "Titel")}
    first = {t: 0 for t in all_teams}
    top2 = {t: 0 for t in all_teams}
    sum_pts = {t: 0.0 for t in all_teams}
    sum_gd = {t: 0.0 for t in all_teams}

    # Memoisierte K.-o.-Weiterkommens-Wahrscheinlichkeit (neutraler Platz).
    adv_cache: Dict[Tuple[str, str], float] = {}

    def adv(home: str, away: str) -> float:
        key = (home, away)
        if key not in adv_cache:
            # K.-o.-Spiele auf neutralem Platz (kein Heimvorteil).
            ph, pdr, pa, _lh, _la = predictor.matchup_from_states(home, away, 1, prepared.states)
            adv_cache[key] = _advance_prob(ph, pdr, pa)
        return adv_cache[key]

    later_order = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, FINAL]

    for _ in range(n):
        winners: Dict[str, str] = {}
        runners: Dict[str, str] = {}
        thirds: List[Tuple[str, str, float, float, float]] = []

        for g, games in grids.items():
            pts: Dict[str, float] = {}
            gd: Dict[str, int] = {}
            gf: Dict[str, int] = {}
            for home, away, cum, kp1 in games:
                for t in (home, away):
                    pts.setdefault(t, 0.0); gd.setdefault(t, 0); gf.setdefault(t, 0)
                idx = int(np.searchsorted(cum, rng.random()))
                hg, ag = divmod(idx, kp1)
                gf[home] += hg; gf[away] += ag
                gd[home] += hg - ag; gd[away] += ag - hg
                if hg > ag:
                    pts[home] += 3
                elif hg < ag:
                    pts[away] += 3
                else:
                    pts[home] += 1; pts[away] += 1
            order = sorted(pts.keys(), key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
            winners[g] = order[0]
            runners[g] = order[1]
            thirds.append((order[2], g, pts[order[2]], gd[order[2]], gf[order[2]]))
            first[order[0]] += 1
            top2[order[0]] += 1
            top2[order[1]] += 1
            for t in pts:
                sum_pts[t] += pts[t]
                sum_gd[t] += gd[t]

        best = sorted(thirds, key=lambda r: (r[2], r[3], r[4], rng.random()), reverse=True)[:8]
        third_by_group = {g: team for (team, g, *_r) in best}
        third_assign = _assign_thirds([g for (_t, g, *_r) in best])
        pairs = _build_r32(winners, runners, third_by_group, third_assign)

        # Sechzehntelfinale erreicht: alle 32.
        win: Dict[int, str] = {}
        for mid, (home, away) in pairs.items():
            reach["R32"][home] += 1
            reach["R32"][away] += 1
            win[mid] = home if rng.random() < adv(home, away) else away

        round_reached = {89: "R16", 97: "QF", 101: "SF", FINAL: "Titel"}
        for mid in later_order:
            (_, ch), (_, ca) = LATER[mid]
            home, away = win[ch], win[ca]
            # "erreicht Runde X" = steht in einem Spiel dieser Runde.
            rnd_in = ROUND_OF[mid]
            reach[rnd_in][home] += 1
            reach[rnd_in][away] += 1
            win[mid] = home if rng.random() < adv(home, away) else away
        reach["Titel"][win[FINAL]] += 1

    # --- Wahrscheinlichkeiten zusammenstellen ------------------------------
    rows = []
    for t in all_teams:
        rows.append({
            "team": t, "group": group_of[t],
            "P(R32)": round(reach["R32"][t] / n, 3),
            "P(R16)": round(reach["R16"][t] / n, 3),
            "P(QF)": round(reach["QF"][t] / n, 3),
            "P(SF)": round(reach["SF"][t] / n, 3),
            "P(Finale)": round(reach["Final"][t] / n, 3),
            "P(Titel)": round(reach["Titel"][t] / n, 3),
        })
    probs = pd.DataFrame(rows).sort_values(["P(Titel)", "P(Finale)", "P(SF)"], ascending=False).reset_index(drop=True)
    probs.insert(0, "rank", range(1, len(probs) + 1))

    grows = []
    for t in all_teams:
        grows.append({
            "group": group_of[t], "team": t,
            "P(Platz 1)": round(first[t] / n, 3),
            "P(Top 2)": round(top2[t] / n, 3),
            "P(weiter)": round(reach["R32"][t] / n, 3),
            "Ø Punkte": round(sum_pts[t] / n, 2),
            "Ø Tordiff": round(sum_gd[t] / n, 2),
        })
    group_df = pd.DataFrame(grows).sort_values(["group", "P(Top 2)"], ascending=[True, False]).reset_index(drop=True)

    return TournamentResult(probs=probs, groups=group_df)
