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
    predictor.save(MODEL_PATH)
    print(f"Modell gespeichert unter {MODEL_PATH}\n")

    print("=== Trainings-Zusammenfassung ===")
    print(json.dumps(predictor.training_summary, indent=2, ensure_ascii=False))

    if len(matches) >= 500:
        print("\n=== Out-of-Sample-Backtest ===")
        print(json.dumps(run_backtest(matches), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
