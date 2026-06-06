"""WM-2026 K.-o.-Phase: offizielles Bracket + Turnier-Simulation + bester Baum.

Die WM 2026 hat ein neues Format: 12 Gruppen a 4 Teams. Weiter kommen die
**zwei Bestplatzierten je Gruppe** (24) plus die **acht besten Gruppendritten**
(8) -> 32 Teams in einer reinen K.-o.-Runde (Achtundsechzigstel… nein:
Sechzehntelfinale/Round of 32 -> Achtel -> Viertel -> Halb -> Finale).

Das Bracket-Skelett (welcher Gruppensieger/-zweite/-dritte in welchem Spiel
antritt) ist von der FIFA **vorab festgelegt** und hier 1:1 hinterlegt
(Spiele 73-104 des offiziellen Spielplans). Nur die konkrete Zuordnung der
acht Gruppendritten haengt davon ab, aus welchen Gruppen sie kommen - das
regelt die FIFA ueber eine feste Kombinationstabelle (Annex C, 495 Faelle).
Wir bilden diese Zuordnung ueber ein zulaessiges Matching der erlaubten
Gruppen-Sets nach (gleiche Bedingungen: ein Dritter trifft nie auf seinen
eigenen Gruppensieger, jede Slot-Belegung respektiert das erlaubte Set).

> Ehrlichkeit: Das Skelett entspricht dem offiziellen Spielplan. In seltenen
> mehrdeutigen Drittel-Konstellationen kann unsere deterministische Zuordnung
> minimal von der FIFA-Tabelle abweichen - sie bleibt aber stets **regelkonform**
> (richtiges Set, kein Gruppen-Rematch) und liefert ein vollstaendiges Bracket.

K.-o.-Spiele kennen kein Remis: Aus der 1X2-Vorhersage wird eine
Weiterkommens-Wahrscheinlichkeit, indem der Remis-Anteil ueber die relative
Staerke (Verlaengerung/Elfmeter) auf beide Teams aufgeteilt wird.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .teams import normalize_team

# --------------------------------------------------------------------------
# Offizielles Bracket (FIFA-Spielplan, Spiele 73-104).
#
# Slot-Notation:
#   ("W", "A")   -> Sieger Gruppe A          (1A)
#   ("R", "A")   -> Zweiter Gruppe A         (2A)
#   ("3", "ABCDF") -> bester Dritter aus dem erlaubten Gruppen-Set {A,B,C,D,F}
#   ("M", 73)    -> Sieger aus Spiel 73 (spaetere Runden)
# --------------------------------------------------------------------------
R32: Dict[int, Tuple[tuple, tuple]] = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("3", "ABCDF")),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("3", "CDFGH")),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("3", "CEFHI")),
    80: (("W", "L"), ("3", "EHIJK")),
    81: (("W", "D"), ("3", "BEFIJ")),
    82: (("W", "G"), ("3", "AEHIJ")),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("3", "EFGIJ")),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("3", "DEIJL")),
    88: (("R", "D"), ("R", "G")),
}

# Spaetere Runden: Spiel-ID -> (Quelle Heim, Quelle Gast) als ("M", id).
LATER: Dict[int, Tuple[tuple, tuple]] = {
    # Achtelfinale (Round of 16)
    89: (("M", 74), ("M", 77)),
    90: (("M", 73), ("M", 75)),
    91: (("M", 76), ("M", 78)),
    92: (("M", 79), ("M", 80)),
    93: (("M", 83), ("M", 84)),
    94: (("M", 81), ("M", 82)),
    95: (("M", 86), ("M", 88)),
    96: (("M", 85), ("M", 87)),
    # Viertelfinale
    97: (("M", 89), ("M", 90)),
    98: (("M", 93), ("M", 94)),
    99: (("M", 91), ("M", 92)),
    100: (("M", 95), ("M", 96)),
    # Halbfinale
    101: (("M", 97), ("M", 98)),
    102: (("M", 99), ("M", 100)),
    # Finale
    104: (("M", 101), ("M", 102)),
}

# Runden-Zuordnung je Spiel-ID (fuer Anzeige + Reihenfolge).
ROUND_OF: Dict[int, str] = (
    {m: "R32" for m in range(73, 89)}
    | {m: "R16" for m in range(89, 97)}
    | {m: "QF" for m in range(97, 101)}
    | {101: "SF", 102: "SF", 104: "Final"}
)
ROUND_ORDER = ["R32", "R16", "QF", "SF", "Final"]
ROUND_LABEL = {
    "R32": "Sechzehntelfinale", "R16": "Achtelfinale", "QF": "Viertelfinale",
    "SF": "Halbfinale", "Final": "Finale",
}

# K.-o.-Termine (offizieller Spielplan) - nur fuer die Anzeige.
KO_DATES: Dict[int, str] = {
    73: "2026-06-28", 74: "2026-06-29", 75: "2026-06-29", 76: "2026-06-29",
    77: "2026-06-30", 78: "2026-06-30", 79: "2026-06-30", 80: "2026-07-01",
    81: "2026-07-01", 82: "2026-07-01", 83: "2026-07-02", 84: "2026-07-02",
    85: "2026-07-02", 86: "2026-07-03", 87: "2026-07-03", 88: "2026-07-03",
    89: "2026-07-04", 90: "2026-07-04", 91: "2026-07-05", 92: "2026-07-05",
    93: "2026-07-06", 94: "2026-07-06", 95: "2026-07-07", 96: "2026-07-07",
    97: "2026-07-09", 98: "2026-07-10", 99: "2026-07-11", 100: "2026-07-11",
    101: "2026-07-14", 102: "2026-07-15", 104: "2026-07-19",
}

# Slots der acht Gruppendritten -> erlaubtes Gruppen-Set.
THIRD_SLOTS: Dict[int, frozenset] = {
    mid: frozenset(slot[1])
    for mid, (h, a) in R32.items()
    for slot in (h, a) if slot[0] == "3"
}


def assign_thirds(qualified_groups: List[str]) -> Dict[int, str]:
    """Ordnet die acht qualifizierten Gruppendritten den acht Slots zu.

    ``qualified_groups``: Liste der Gruppen-Buchstaben, deren Dritter es unter
    die besten acht geschafft hat (Laenge 8). Rueckgabe: ``{spiel_id: gruppe}``.

    Loest ein perfektes Matching (jeder Slot bekommt genau eine erlaubte
    Gruppe). Deterministisch via Backtracking, Slots mit den wenigsten Optionen
    zuerst (regelkonform zur FIFA-Vorgabe; siehe Modul-Docstring).
    """
    groups = set(qualified_groups)
    slots = list(THIRD_SLOTS.items())  # (mid, erlaubtes set)

    def backtrack(remaining_slots, used):
        if not remaining_slots:
            return {}
        # Most-constrained-first: Slot mit den wenigsten verbleibenden Optionen.
        remaining_slots = sorted(
            remaining_slots,
            key=lambda ms: len((ms[1] & groups) - used),
        )
        mid, allowed = remaining_slots[0]
        rest = remaining_slots[1:]
        for g in sorted((allowed & groups) - used):  # alphabetisch -> deterministisch
            sub = backtrack(rest, used | {g})
            if sub is not None:
                sub[mid] = g
                return sub
        return None

    result = backtrack(slots, set())
    if result is None:
        # Fallback (sollte bei gueltigen 8 Gruppen nie noetig sein): greedy.
        result = {}
        used: set = set()
        for mid, allowed in sorted(slots, key=lambda ms: len(ms[1] & groups)):
            for g in sorted((allowed & groups) - used):
                result[mid] = g
                used.add(g)
                break
    return result


# --------------------------------------------------------------------------
# Weiterkommens-Wahrscheinlichkeit fuer ein K.-o.-Spiel.
# --------------------------------------------------------------------------
def advance_prob(p_home: float, p_draw: float, p_away: float) -> float:
    """P(Heim kommt weiter) in einem K.-o.-Spiel (Remis -> Verlaengerung/Elfer).

    Der Remis-Anteil wird proportional zur relativen Siegwahrscheinlichkeit
    aufgeteilt - das staerkere Team gewinnt die Entscheidung etwas haeufiger,
    aber nicht deterministisch (Elfmeterschiessen ist nahe am Muenzwurf).
    """
    denom = p_home + p_away
    share = 0.5 if denom <= 1e-9 else p_home / denom
    return float(np.clip(p_home + p_draw * share, 0.0, 1.0))


# --------------------------------------------------------------------------
# Turnier-Simulation (Monte Carlo): Gruppen + K.-o. in einem Durchlauf.
# --------------------------------------------------------------------------
@dataclass
class TournamentResult:
    probs: pd.DataFrame                 # je Team: P(R16/QF/SF/Final/Titel)
    n: int
    champion: str = ""                  # haeufigster Sieger
    groups: pd.DataFrame | None = None  # je Team: P(Platz 1/2/Top 2) in der Gruppe
    field: Dict[str, object] = field(default_factory=dict)


def _adv_matrix(predictor, states, teams: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Neutrale Weiterkommens-Matrix adv[i,j] = P(Team i schlaegt Team j)."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    adv = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            ph, pd_, pa, _, _ = predictor.matchup_from_states(teams[i], teams[j], 1, states)
            a = advance_prob(ph, pd_, pa)
            adv[i, j] = a
            adv[j, i] = 1.0 - a
    return adv, idx


@dataclass
class Prepared:
    """Vorberechnete Turnierdaten (teuer; einmal je Modell-/Kader-Konfiguration)."""
    teams: List[str]
    states: Dict
    adv: np.ndarray
    idx: Dict[str, int]
    group_of: Dict[str, str]


def prepare_tournament(predictor, matches: pd.DataFrame, groups: Dict[str, List[str]]) -> Prepared:
    """Berechnet Team-Zustaende + paarweise Weiterkommens-Matrix (einmalig).

    Diese Vorbereitung dominiert die Laufzeit; sowohl die Monte-Carlo-Simulation
    als auch der wahrscheinlichste Baum koennen das Ergebnis wiederverwenden.
    """
    teams = [normalize_team(t) for g in groups.values() for t in g]
    group_of = {normalize_team(t): g for g, ts in groups.items() for t in ts}
    states = predictor.prepare_states(matches, teams)
    adv, idx = _adv_matrix(predictor, states, teams)
    return Prepared(teams=teams, states=states, adv=adv, idx=idx, group_of=group_of)


def _group_fixture_samples(predictor, states, fixtures: pd.DataFrame, n: int, rng):
    """Zieht je Gruppenspiel n Korrektergebnisse vorab (vektorisiert).

    Rueckgabe: Liste je Gruppe -> Liste von (heim, gast, hg_array, ag_array).
    """
    by_group: Dict[str, list] = {}
    for _, fx in fixtures.iterrows():
        home = normalize_team(fx["home_team"])
        away = normalize_team(fx["away_team"])
        neutral = int(fx.get("neutral", 1) or 0)
        _, _, _, lh, la = predictor.matchup_from_states(home, away, neutral, states)
        grid = predictor.score_grid_from_lams(lh, la)
        flat = grid.flatten()
        flat = flat / flat.sum()
        kdim = grid.shape[0]
        draws = rng.choice(len(flat), size=n, p=flat)
        hg, ag = np.divmod(draws, kdim)
        by_group.setdefault(str(fx["group_name"]), []).append((home, away, hg, ag))
    return by_group


def simulate_tournament(
    predictor,
    matches: pd.DataFrame,
    groups: Dict[str, List[str]],
    fixtures: pd.DataFrame | None = None,
    n: int = 2000,
    seed: int = 20260611,
    prepared: Prepared | None = None,
) -> TournamentResult:
    """Simuliert das komplette Turnier (Gruppen -> Finale) n-mal.

    Liefert je Team die Wahrscheinlichkeit, jede Runde zu erreichen, sowie den
    Turniersieg. Effizient: Team-Zustaende und paarweise Wahrscheinlichkeiten
    werden **einmal** vorberechnet (oder via ``prepared`` wiederverwendet),
    danach ist jeder Durchlauf reines Sampling.
    """
    from .wm2026 import build_fixtures

    if fixtures is None:
        fixtures = build_fixtures()
    if prepared is None:
        prepared = prepare_tournament(predictor, matches, groups)
    teams, states, adv, idx, group_of = (
        prepared.teams, prepared.states, prepared.adv, prepared.idx, prepared.group_of
    )
    rng = np.random.default_rng(seed)

    group_samples = _group_fixture_samples(predictor, states, fixtures, n, rng)

    # Zaehler je Team.
    reach = {r: np.zeros(len(teams)) for r in ["R32", "R16", "QF", "SF", "Final", "Title"]}
    first_cnt = np.zeros(len(teams))
    second_cnt = np.zeros(len(teams))

    # Jitter fuer reproduzierbare, faire Tiebreaks.
    jitter = rng.random((n, len(teams))) * 1e-6

    for s in range(n):
        # ---- Gruppenphase: Tabellen rechnen --------------------------------
        firsts, seconds = {}, {}
        thirds = []  # (group, pts, gd, gf, team)
        for g, gteams in groups.items():
            gt = [normalize_team(t) for t in gteams]
            pts = {t: 0 for t in gt}
            gd = {t: 0 for t in gt}
            gf = {t: 0 for t in gt}
            for home, away, hg_arr, ag_arr in group_samples[g]:
                hg, ag = int(hg_arr[s]), int(ag_arr[s])
                gf[home] += hg
                gf[away] += ag
                gd[home] += hg - ag
                gd[away] += ag - hg
                if hg > ag:
                    pts[home] += 3
                elif ag > hg:
                    pts[away] += 3
                else:
                    pts[home] += 1
                    pts[away] += 1
            order = sorted(gt, key=lambda t: (pts[t], gd[t], gf[t], jitter[s, idx[t]]), reverse=True)
            firsts[g] = order[0]
            seconds[g] = order[1]
            first_cnt[idx[order[0]]] += 1
            second_cnt[idx[order[1]]] += 1
            thirds.append((g, pts[order[2]], gd[order[2]], gf[order[2]], order[2]))

        # ---- beste acht Dritte ---------------------------------------------
        thirds.sort(key=lambda x: (x[1], x[2], x[3], jitter[s, idx[x[4]]]), reverse=True)
        best = thirds[:8]
        third_team_of_group = {g: t for g, _, _, _, t in best}
        slot_group = assign_thirds([g for g, *_ in best])

        # Qualifiziert (R32 erreicht): 24 + 8.
        for t in list(firsts.values()) + list(seconds.values()) + [t for *_, t in best]:
            reach["R32"][idx[t]] += 1

        # ---- Bracket fuellen ------------------------------------------------
        def slot_team(slot, mid):
            kind, val = slot
            if kind == "W":
                return firsts[val]
            if kind == "R":
                return seconds[val]
            # Dritter: welche Gruppe wurde diesem Slot zugeordnet?
            return third_team_of_group[slot_group[mid]]

        winners: Dict[int, str] = {}
        # Round of 32.
        for mid, (hs_, as_) in R32.items():
            h = slot_team(hs_, mid)
            a = slot_team(as_, mid)
            winners[mid] = _play(adv, idx, h, a, rng)
        # Spaetere Runden in fester Reihenfolge.
        for mid in sorted(LATER):
            (kh, vh), (ka, va) = LATER[mid]
            h, a = winners[vh], winners[va]
            winners[mid] = _play(adv, idx, h, a, rng)

        # ---- Runden gutschreiben -------------------------------------------
        # Erreichte Runde = Sieger des vorigen Spiels.
        for mid in range(73, 89):  # Sieger R32 -> Achtelfinale erreicht
            reach["R16"][idx[winners[mid]]] += 1
        for mid in range(89, 97):  # Sieger Achtel -> Viertelfinale
            reach["QF"][idx[winners[mid]]] += 1
        for mid in range(97, 101):  # Sieger Viertel -> Halbfinale
            reach["SF"][idx[winners[mid]]] += 1
        for mid in (101, 102):      # Sieger Halb -> Finale
            reach["Final"][idx[winners[mid]]] += 1
        reach["Title"][idx[winners[104]]] += 1

    rows = []
    for t in teams:
        i = idx[t]
        rows.append({
            "team": t,
            "group": group_of[t],
            "P(R32)": round(reach["R32"][i] / n, 3),
            "P(R16)": round(reach["R16"][i] / n, 3),
            "P(QF)": round(reach["QF"][i] / n, 3),
            "P(SF)": round(reach["SF"][i] / n, 3),
            "P(Finale)": round(reach["Final"][i] / n, 3),
            "P(Titel)": round(reach["Title"][i] / n, 3),
        })
    df = pd.DataFrame(rows).sort_values("P(Titel)", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    champ = df.iloc[0]["team"] if len(df) else ""

    grp_rows = []
    for t in teams:
        i = idx[t]
        p1, p2 = first_cnt[i] / n, second_cnt[i] / n
        grp_rows.append({
            "group": group_of[t], "team": t,
            "P(Platz 1)": round(p1, 3), "P(Platz 2)": round(p2, 3),
            "P(Top 2)": round(p1 + p2, 3), "P(weiter)": round(reach["R32"][i] / n, 3),
        })
    groups_df = (pd.DataFrame(grp_rows)
                 .sort_values(["group", "P(Top 2)"], ascending=[True, False])
                 .reset_index(drop=True))
    return TournamentResult(probs=df, n=n, champion=champ, groups=groups_df)


def _play(adv: np.ndarray, idx: Dict[str, int], home: str, away: str, rng) -> str:
    """Ein K.-o.-Spiel: Sieger per Bernoulli aus der Weiterkommens-Matrix."""
    p = adv[idx[home], idx[away]]
    return home if rng.random() < p else away


# --------------------------------------------------------------------------
# Wahrscheinlichster Turnierbaum (deterministisch, fuer die Anzeige).
# --------------------------------------------------------------------------
@dataclass
class BracketMatch:
    match_id: int
    round: str
    date: str
    home: str
    away: str
    p_home: float          # P(Heim kommt weiter)
    winner: str


def most_likely_bracket(
    predictor,
    matches: pd.DataFrame,
    groups: Dict[str, List[str]],
    fixtures: pd.DataFrame | None = None,
    prepared: Prepared | None = None,
) -> Tuple[List[BracketMatch], pd.DataFrame]:
    """Baut den wahrscheinlichsten Turnierbaum aus den Erwartungswerten.

    Gruppen werden ueber **erwartete Punkte** (aus den paarweisen 1X2-Wahr-
    scheinlichkeiten der echten Gruppenspiele) sortiert; in jedem K.-o.-Spiel
    kommt der jeweilige Favorit weiter. Rueckgabe: Liste der Spiele (R32..Finale)
    plus die ermittelten Gruppentabellen.
    """
    from .wm2026 import build_fixtures

    if fixtures is None:
        fixtures = build_fixtures()
    if prepared is None:
        prepared = prepare_tournament(predictor, matches, groups)
    states, adv, idx = prepared.states, prepared.adv, prepared.idx
    teams = prepared.teams

    # Erwartete Punkte/Tordifferenz je Team aus den echten Gruppenspielen.
    exp_pts = {t: 0.0 for t in teams}
    exp_gd = {t: 0.0 for t in teams}
    for _, fx in fixtures.iterrows():
        home = normalize_team(fx["home_team"])
        away = normalize_team(fx["away_team"])
        neutral = int(fx.get("neutral", 1) or 0)
        ph, pdr, pa, lh, la = predictor.matchup_from_states(home, away, neutral, states)
        exp_pts[home] += 3 * ph + pdr
        exp_pts[away] += 3 * pa + pdr
        exp_gd[home] += lh - la
        exp_gd[away] += la - lh

    standings_rows = []
    firsts, seconds, thirds = {}, {}, []
    for g, gteams in groups.items():
        gt = [normalize_team(t) for t in gteams]
        order = sorted(gt, key=lambda t: (exp_pts[t], exp_gd[t]), reverse=True)
        firsts[g], seconds[g] = order[0], order[1]
        thirds.append((g, exp_pts[order[2]], exp_gd[order[2]], order[2]))
        for pos, t in enumerate(order, 1):
            standings_rows.append({
                "group": g, "pos": pos, "team": t,
                "exp_points": round(exp_pts[t], 2), "exp_gd": round(exp_gd[t], 2),
            })
    standings = pd.DataFrame(standings_rows)

    thirds.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = thirds[:8]
    third_team_of_group = {g: t for g, _, _, t in best}
    slot_group = assign_thirds([g for g, *_ in best])

    def slot_team(slot, mid):
        kind, val = slot
        if kind == "W":
            return firsts[val]
        if kind == "R":
            return seconds[val]
        return third_team_of_group[slot_group[mid]]

    out: List[BracketMatch] = []
    winners: Dict[int, str] = {}
    for mid, (hs_, as_) in R32.items():
        h, a = slot_team(hs_, mid), slot_team(as_, mid)
        p = float(adv[idx[h], idx[a]])
        w = h if p >= 0.5 else a
        winners[mid] = w
        out.append(BracketMatch(mid, "R32", KO_DATES.get(mid, ""), h, a, round(p, 3), w))
    for mid in sorted(LATER):
        (_, vh), (_, va) = LATER[mid]
        h, a = winners[vh], winners[va]
        p = float(adv[idx[h], idx[a]])
        w = h if p >= 0.5 else a
        winners[mid] = w
        out.append(BracketMatch(mid, ROUND_OF[mid], KO_DATES.get(mid, ""), h, a, round(p, 3), w))

    out.sort(key=lambda m: m.match_id)
    return out, standings


def bracket_to_frame(bracket: List[BracketMatch]) -> pd.DataFrame:
    """Turnierbaum als flache Tabelle (fuer CSV/Anzeige)."""
    return pd.DataFrame([
        {
            "match_id": m.match_id, "round": m.round, "round_label": ROUND_LABEL[m.round],
            "date": m.date, "home": m.home, "away": m.away,
            "p_home_advance": m.p_home, "winner": m.winner,
        }
        for m in bracket
    ])
