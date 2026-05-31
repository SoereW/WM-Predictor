"""Validierung an echten Spielen mit bekanntem Endergebnis (out-of-sample).

Setzt den gewuenschten Ablauf um:
1. Mannschaften/Spieler vergleichen (Bewertungssystem -> Team-Scores, beste XI).
2. Vorhersage aufstellen (Predictor: 1X2 + erwartete Tore).
3. Mit dem tatsaechlichen Endergebnis vergleichen und auswerten.

EHRLICH = OUT-OF-SAMPLE: Fuer jedes Testspiel wird ein frisches Modell nur
auf Spielen *vor* dem Spieldatum trainiert. Das Modell kennt das Ergebnis
also nicht - es ist eine echte Prognose, kein Nachtippen. (Mit --insample
laesst sich zum Vergleich das fertige Gesamtmodell verwenden.)

Die Testspiele sind echte Laenderspiele aus dem Datensatz zwischen Teams,
fuer die detaillierte Spielerdaten vorliegen (src/rating/sample.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database import connect, read_table
from src.model import WMPredictor
from src.rating.sample import COACHES, load_sample_players
from src.rating.team import rate_all_teams
from src.squad import load_squads

DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
MODEL_PATH = ROOT / "models" / "wm_predictor.joblib"
SQUADS_CSV = ROOT / "data" / "sample_squads.csv"

# Echte Spiele (im Datensatz vorhanden) zwischen Teams mit Spielerdaten.
# (datum, heim, gast, heim_tore, gast_tore, neutral, wettbewerb)
TEST_MATCHES = [
    ("2024-03-23", "France", "Germany", 0, 2, 0, "Friendly"),
    ("2024-10-10", "Saudi Arabia", "Japan", 0, 2, 0, "WM-Quali"),
    ("2025-03-25", "Argentina", "Brazil", 4, 1, 0, "WM-Quali"),
    ("2025-06-08", "Germany", "France", 0, 2, 0, "Nations League"),
    ("2025-10-14", "Japan", "Brazil", 3, 2, 0, "Kirin Cup"),
    ("2026-03-26", "Brazil", "France", 1, 2, 1, "Friendly"),
]


def _outcome(hs: int, as_: int) -> str:
    return "1" if hs > as_ else ("2" if as_ > hs else "X")


def _fit_before(matches, cutoff_date, squads):
    """Trainiert ein Modell nur auf Spielen vor ``cutoff_date`` (kein Leakage)."""
    train = matches[matches["date"] < cutoff_date].reset_index(drop=True)
    predictor = WMPredictor().fit(train)
    if squads is not None:
        predictor.attach_squads(squads)
    return predictor, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Vorhersagen gegen echte Ergebnisse pruefen")
    parser.add_argument("--insample", action="store_true",
                        help="Fertiges Gesamtmodell nutzen (kennt die Spiele) statt out-of-sample")
    args = parser.parse_args()

    con = connect(DB_PATH)
    matches = read_table(con, "matches")
    con.close()
    squads = load_squads(SQUADS_CSV) if SQUADS_CSV.exists() else None

    full_model = WMPredictor.load(MODEL_PATH) if args.insample else None
    team_scores = rate_all_teams(load_sample_players(), coaches=COACHES)

    mode = "IN-SAMPLE (Modell kennt die Spiele)" if args.insample else "OUT-OF-SAMPLE (echte Prognose)"
    print("=" * 92)
    print(f"VALIDIERUNG [{mode}]: Kader-/Spielervergleich + Vorhersage vs. echtes Ergebnis")
    print("=" * 92)

    pred_correct = 0
    rating_correct = 0
    rating_decisive = 0
    log_losses = []

    for date, home, away, hs, as_, neutral, comp in TEST_MATCHES:
        true_out = _outcome(hs, as_)

        # --- Schritt 1: Mannschaften/Spieler vergleichen (Bewertungssystem) ---
        ts_h, ts_a = team_scores.get(home), team_scores.get(away)
        score_diff = (ts_h.overall - ts_a.overall) if (ts_h and ts_a) else 0.0
        rating_fav = "X" if abs(score_diff) < 1.0 else ("1" if score_diff > 0 else "2")

        # --- Schritt 2: Vorhersage (Predictor), out-of-sample trainiert ---
        if args.insample:
            predictor, hist = full_model, matches
        else:
            predictor, hist = _fit_before(matches, date, squads)

        pred = predictor.predict_fixture(
            {"home_team": home, "away_team": away, "neutral": neutral}, hist
        )
        probs = {"1": pred.home_win, "X": pred.draw, "2": pred.away_win}
        pred_out = max(probs, key=probs.get)
        ll = -np.log(max(probs[true_out], 1e-12))
        log_losses.append(ll)

        # --- Schritt 3: Vergleich mit Realitaet ---
        pred_hit = pred_out == true_out
        pred_correct += int(pred_hit)
        if rating_fav != "X":
            rating_decisive += 1
            rating_correct += int(rating_fav == true_out)

        print(f"\n{date} | {comp}{' | neutraler Platz' if neutral else ''}")
        print(f"  {home} vs {away}   ECHTES ERGEBNIS: {hs}:{as_}  ({true_out})")
        if ts_h and ts_a:
            print(f"  [1] Kadervergleich:  {home} {ts_h.overall:.1f} "
                  f"(XI {ts_h.best_xi_score:.1f} | Chemie {ts_h.chemistry.overall:.1f})   vs   "
                  f"{away} {ts_a.overall:.1f} "
                  f"(XI {ts_a.best_xi_score:.1f} | Chemie {ts_a.chemistry.overall:.1f})")
            print(f"      Score-Differenz {score_diff:+.1f}  =>  Favorit laut Bewertung: {rating_fav} "
                  f"({'richtig' if rating_fav == true_out else 'falsch' if rating_fav != 'X' else 'offen'})")
        print(f"  [2] Vorhersage:  1={probs['1']*100:4.1f}%   X={probs['X']*100:4.1f}%   "
              f"2={probs['2']*100:4.1f}%   | erwartete Tore {pred.expected_home_goals:.2f}:{pred.expected_away_goals:.2f}")
        print(f"      Tipp: {pred_out}  ->  {'TREFFER' if pred_hit else 'daneben'}   "
              f"(Modell gab dem echten Ausgang {probs[true_out]*100:.1f}%, LogLoss {ll:.2f})")

    n = len(TEST_MATCHES)
    print("\n" + "=" * 92)
    print("AUSWERTUNG")
    print("=" * 92)
    print(f"  Predictor-Trefferquote (1X2):        {pred_correct}/{n} = {pred_correct/n*100:.0f}%")
    if rating_decisive:
        print(f"  Bewertungssystem-Trefferquote:       {rating_correct}/{rating_decisive} "
              f"= {rating_correct/rating_decisive*100:.0f}%  (nur Spiele mit klarem Favoriten)")
    print(f"  Mittlerer Log-Loss (Predictor):      {np.mean(log_losses):.3f}  "
          f"(niedriger = besser; blindes Raten ~1.10)")
    print(f"  Mittlere Wahrsch. fuers echte Ergebnis: {np.mean([np.exp(-l) for l in log_losses])*100:.1f}%")
    print("\nHinweis: 6 Spiele sind eine kleine Stichprobe - das zeigt Vorgehen und grobe")
    print("Tendenz, keine statistisch belastbare Quote. Fuer die WM werden alle Spiele")
    print("ausgewertet, sobald Kader und Spielplan feststehen.")


if __name__ == "__main__":
    main()
