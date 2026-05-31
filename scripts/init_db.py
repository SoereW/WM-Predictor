"""Baut die SQLite-Datenbank aus CSV-Daten.

Beispiele:
    python scripts/init_db.py                # nutzt vorhandene/Beispiel-Daten
    python scripts/init_db.py --fetch        # laedt echte Historie (martj42)
    python scripts/init_db.py --matches mein_export.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database import connect, init_schema, load_fixtures, load_matches, read_table

DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="WM-Predictor-Datenbank initialisieren")
    parser.add_argument("--fetch", action="store_true", help="Echte Historie von martj42 laden")
    parser.add_argument("--matches", type=str, default=None, help="Pfad zu einer Match-CSV")
    parser.add_argument("--fixtures", type=str, default=None, help="Pfad zu einer Fixtures-CSV")
    args = parser.parse_args()

    # Beispieldaten bei Bedarf erzeugen (werden nicht ins Repo eingecheckt).
    from src.sample_data import write_sample_data

    sample_matches = DATA / "sample_matches.csv"
    sample_fixtures = DATA / "sample_fixtures_2026.csv"
    if not sample_matches.exists() or not sample_fixtures.exists():
        write_sample_data(DATA)

    if args.matches:
        matches_csv = Path(args.matches)
    elif args.fetch:
        from src.ingest import fetch_international_results

        print("Lade historische Ergebnisse (martj42/international_results) ...")
        matches_csv = fetch_international_results(DATA / "international_results.csv")
        print(f"Gespeichert unter {matches_csv}")
    elif (DATA / "international_results.csv").exists():
        matches_csv = DATA / "international_results.csv"
    else:
        matches_csv = sample_matches

    fixtures_csv = Path(args.fixtures) if args.fixtures else sample_fixtures

    con = connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS matches")
    cur.execute("DROP TABLE IF EXISTS fixtures")
    con.commit()
    init_schema(con)

    n_matches = load_matches(con, matches_csv)
    n_fixtures = load_fixtures(con, fixtures_csv)
    teams = read_table(con, "matches")
    n_teams = len(set(teams["home_team"]) | set(teams["away_team"]))
    con.close()

    print(f"Datenbank: {DB_PATH}")
    print(f"  Matches : {n_matches}  ({matches_csv.name})")
    print(f"  Fixtures: {n_fixtures}")
    print(f"  Teams   : {n_teams}")


if __name__ == "__main__":
    main()
