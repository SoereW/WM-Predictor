"""Trainiert das Modell, speichert es und gibt Qualitaetsmetriken aus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.database import connect, read_table
from src.model import WMPredictor

DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
MODEL_PATH = ROOT / "models" / "wm_predictor.joblib"


def main() -> None:
    con = connect(DB_PATH)
    matches = read_table(con, "matches")
    con.close()

    if matches.empty:
        raise SystemExit("Keine Spiele in der DB. Erst 'python scripts/init_db.py' ausfuehren.")

    print(f"Trainiere auf {len(matches)} Spielen ...")
    predictor = WMPredictor().fit(matches)

    # Bewertungssystem (Spieler + Chemie + Trainer) als Staerkequelle
    # anbinden - reichhaltiger als der reine Kaderdurchschnitt. Faellt auf
    # die einfache Kaderstaerke zurueck, falls das Rating-System keine Daten
    # hat.
    try:
        from src.rating.sample import COACHES, load_sample_players
        from src.rating.team import rate_all_teams

        team_scores = rate_all_teams(load_sample_players(), coaches=COACHES)
        predictor.attach_team_scores(team_scores)
        sq = predictor.training_summary.get("squad", {})
        print(f"Bewertungssystem angebunden: {sq.get('n_teams_with_squad')} Teams, "
              f"kalibriert={sq.get('calibrated')} (Quelle: {sq.get('source')})")
    except Exception as err:  # noqa: BLE001 - robust gegen fehlende Rating-Daten
        squads_csv = ROOT / "data" / "sample_squads.csv"
        if squads_csv.exists():
            from src.squad import load_squads

            predictor.attach_squads(load_squads(squads_csv))
            print(f"Fallback Kaderstaerke angebunden ({err}).")

    predictor.save(MODEL_PATH)
    print(f"Modell gespeichert unter {MODEL_PATH}\n")

    print("=== Trainings-Zusammenfassung ===")
    print(json.dumps(predictor.training_summary, indent=2, ensure_ascii=False))

    if len(matches) >= 500:
        print("\n=== Out-of-Sample-Backtest ===")
        print(json.dumps(run_backtest(matches), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
