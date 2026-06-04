"""Schreibt die mitgelieferten Beispieldaten als CSV.

Duenner Wrapper um ``src.sample_data`` (dort steht die ganze Logik, damit
auch ``init_db.py`` und das Dashboard die Daten bei Bedarf erzeugen
koennen). Details/Annahmen siehe Docstring in ``src/sample_data.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sample_data import GROUPS, write_sample_data


def main() -> None:
    matches_path, fixtures_path = write_sample_data(ROOT / "data")
    import pandas as pd

    n_matches = len(pd.read_csv(matches_path))
    n_fixtures = len(pd.read_csv(fixtures_path))
    print(f"{matches_path.name}      : {n_matches} Spiele")
    print(f"{fixtures_path.name}: {n_fixtures} Fixtures, {len(GROUPS)} Gruppen")


if __name__ == "__main__":
    main()
