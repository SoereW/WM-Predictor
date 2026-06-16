"""Echte WM-2026-Ergebnisse einpflegen und Ergebnis-Tabelle ausgeben.

Die WM 2026 laeuft bereits. Dieses Skript verknuepft die **tatsaechlichen
Endergebnisse** der bisher gespielten Gruppenspiele (zentral gepflegt in
``src/wm2026_live.py``) mit dem Spielplan (``data/wm2026_fixtures.csv``) und
schreibt eine vollstaendige Ergebnis-Tabelle nach ``data/wm2026_results.csv``.

Aktualisieren: neue Eintraege in ``src/wm2026_live.ACTUAL_RESULTS`` ergaenzen
(Schluessel = ``match_id``, Wert = ``(heim_tore, gast_tore)``) und das Skript
erneut laufen lassen. Dieselben Ergebnisse fliessen ueber
``scripts/build_wm2026.py`` in die Teamstaerken und die Vorhersagen ein.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wm2026_live import ACTUAL_RESULTS, AS_OF, outcome as _outcome

FIXTURES_CSV = ROOT / "data" / "wm2026_fixtures.csv"
PREDICTIONS_CSV = ROOT / "data" / "wm2026_predictions.csv"
RESULTS_CSV = ROOT / "data" / "wm2026_results.csv"


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
