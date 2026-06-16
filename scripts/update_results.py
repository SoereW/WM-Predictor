"""Echte WM-2026-Ergebnisse einpflegen und Ergebnis-Tabelle ausgeben.

Die WM 2026 laeuft bereits. Dieses Skript haelt die **tatsaechlichen
Endergebnisse** der bisher gespielten Gruppenspiele (Stand siehe ``AS_OF``),
verknuepft sie mit dem Spielplan (``data/wm2026_fixtures.csv``) und schreibt
eine vollstaendige Ergebnis-Tabelle nach ``data/wm2026_results.csv``.

Aktualisieren: einfach neue Eintraege in ``ACTUAL_RESULTS`` ergaenzen
(Schluessel = ``match_id`` aus dem Spielplan, Wert = ``(heim_tore, gast_tore)``)
und das Skript erneut laufen lassen.

Quellen (Web-Recherche, Stand 16.06.2026): CBS Sports, Yahoo Sports,
SBS News, Sky Sports, ESPN, Wikipedia (2026 FIFA World Cup group stage).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Datum, bis zu dem die Ergebnisse vollstaendig sind.
AS_OF = "2026-06-16"

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_CSV = ROOT / "data" / "wm2026_fixtures.csv"
PREDICTIONS_CSV = ROOT / "data" / "wm2026_predictions.csv"
RESULTS_CSV = ROOT / "data" / "wm2026_results.csv"

# Tatsaechliche Endergebnisse: match_id -> (heim_tore, gast_tore).
# Die Orientierung (Heim/Gast) entspricht dem Spielplan in wm2026_fixtures.csv.
# Bisher gespielt: 1. Spieltag der Gruppen A-H (16 Spiele). Die Partien vom
# 16.06. (Gruppen I-L) stossen erst am Abend (ET) an und fehlen daher noch.
ACTUAL_RESULTS: dict[int, tuple[int, int]] = {
    # Gruppe A
    1: (2, 0),   # Mexico 2-0 South Africa
    2: (2, 1),   # South Korea 2-1 Czech Republic
    # Gruppe B
    7: (1, 1),   # Canada 1-1 Bosnia-Herzegovina
    8: (1, 1),   # Qatar 1-1 Switzerland
    # Gruppe C
    13: (1, 1),  # Brazil 1-1 Morocco
    14: (0, 1),  # Haiti 0-1 Scotland
    # Gruppe D
    19: (4, 1),  # United States 4-1 Paraguay
    20: (2, 0),  # Australia 2-0 Turkey
    # Gruppe E
    25: (7, 1),  # Germany 7-1 Curacao
    26: (1, 0),  # Ivory Coast 1-0 Ecuador
    # Gruppe F
    31: (2, 2),  # Netherlands 2-2 Japan
    32: (5, 1),  # Sweden 5-1 Tunisia
    # Gruppe G
    37: (1, 1),  # Belgium 1-1 Egypt
    38: (2, 2),  # Iran 2-2 New Zealand
    # Gruppe H
    43: (0, 0),  # Spain 0-0 Cape Verde
    44: (1, 1),  # Saudi Arabia 1-1 Uruguay
}


def _outcome(hs: int, as_: int) -> str:
    """1 = Heimsieg, X = Unentschieden, 2 = Auswaertssieg."""
    if hs > as_:
        return "1"
    if hs < as_:
        return "2"
    return "X"


def _load_tips() -> dict[tuple[str, str], str]:
    """Modell-Tipp (1/X/2) je Partie aus den Vorhersagen, falls vorhanden."""
    if not PREDICTIONS_CSV.exists():
        return {}
    pred = pd.read_csv(PREDICTIONS_CSV)
    return {
        (r["home_team"], r["away_team"]): str(r["tip"])
        for _, r in pred.iterrows()
    }


def build_results() -> pd.DataFrame:
    """Spielplan + tatsaechliche Ergebnisse zu einer Tabelle verbinden."""
    fx = pd.read_csv(FIXTURES_CSV)
    tips = _load_tips()

    rows = []
    for _, m in fx.iterrows():
        mid = int(m["match_id"])
        tip = tips.get((m["home_team"], m["away_team"]), "")
        score = ACTUAL_RESULTS.get(mid)
        if score is not None:
            hs, as_ = score
            result = _outcome(hs, as_)
            rows.append({
                "match_id": mid,
                "group": m["group_name"],
                "date": m["date"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_score": hs,
                "away_score": as_,
                "score": f"{hs}:{as_}",
                "result": result,
                "status": "played",
                "tip": tip,
                "tip_hit": "" if not tip else ("1" if tip == result else "0"),
            })
        else:
            rows.append({
                "match_id": mid,
                "group": m["group_name"],
                "date": m["date"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_score": pd.NA,
                "away_score": pd.NA,
                "score": "-:-",
                "result": "",
                "status": "scheduled",
                "tip": tip,
                "tip_hit": "",
            })

    df = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)
    return df


def print_table(df: pd.DataFrame) -> None:
    """Alle Spiele der Reihe nach mit Spielausgang ausgeben."""
    played = int((df["status"] == "played").sum())
    print(f"WM 2026 - Ergebnisse (Stand {AS_OF}): "
          f"{played}/{len(df)} Spiele gespielt\n")

    header = f"{'#':>3}  {'Gr':<2}  {'Datum':<10}  " \
             f"{'Heim':>20}  {'Erg':^5}  {'Gast':<20}  {'Ausgang':<7}  {'Tipp'}"
    print(header)
    print("-" * len(header))

    last_group = None
    for _, r in df.iterrows():
        if last_group is not None and r["group"] != last_group:
            print()
        last_group = r["group"]
        if r["status"] == "played":
            outcome = r["result"]
            if r["tip_hit"] == "1":
                tip = f"{r['tip']} (Treffer)"
            elif r["tip_hit"] == "0":
                tip = f"{r['tip']} (daneben)"
            else:
                tip = r["tip"]
        else:
            outcome = "offen"
            tip = f"{r['tip']} (Prognose)" if r["tip"] else ""
        print(f"{r['match_id']:>3}  {r['group']:<2}  {r['date']:<10}  "
              f"{r['home_team']:>20}  {r['score']:^5}  {r['away_team']:<20}  "
              f"{outcome:<7}  {tip}")

    # Trefferquote des Modells auf den bereits gespielten Spielen.
    done = df[df["status"] == "played"]
    scored = done[done["tip_hit"] != ""]
    if len(scored):
        hits = int((scored["tip_hit"] == "1").sum())
        print(f"\nModell-Tipp: {hits}/{len(scored)} Spiele richtig "
              f"({hits / len(scored):.0%}).")


def main() -> None:
    df = build_results()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print_table(df)
    print(f"\nGeschrieben: {RESULTS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
