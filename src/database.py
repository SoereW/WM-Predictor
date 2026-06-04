"""SQLite-Datenmodell und CSV-Ingestion.

Schema bewusst kompatibel zum offenen Datensatz
``martj42/international_results`` gehalten (Spalten
``home_score``/``away_score``), damit echte historische Daten ohne
Umbau geladen werden koennen.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .teams import normalize_team

MATCHES_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    date        TEXT NOT NULL,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_score  INTEGER NOT NULL,
    away_score  INTEGER NOT NULL,
    tournament  TEXT,
    city        TEXT,
    country     TEXT,
    neutral     INTEGER DEFAULT 0
);
"""

FIXTURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS fixtures (
    match_id    INTEGER PRIMARY KEY,
    date        TEXT NOT NULL,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    city        TEXT,
    country     TEXT,
    neutral     INTEGER DEFAULT 1,
    group_name  TEXT,
    stage       TEXT
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_team);",
    "CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_team);",
]


def connect(db_path) -> sqlite3.Connection:
    """Oeffnet (und legt bei Bedarf an) die SQLite-Datenbank."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def init_schema(con: sqlite3.Connection) -> None:
    """Legt Tabellen und Indizes an."""
    cur = con.cursor()
    cur.execute(MATCHES_SCHEMA)
    cur.execute(FIXTURES_SCHEMA)
    for stmt in INDEXES:
        cur.execute(stmt)
    con.commit()


def _coerce_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Vereinheitlicht Spaltennamen/Typen eines Match-DataFrames."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    renames = {
        "home_goals": "home_score",
        "away_goals": "away_score",
        "home goals": "home_score",
        "away goals": "away_score",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Match-CSV fehlt Spalten: {sorted(missing)}")

    for opt in ("tournament", "city", "country"):
        if opt not in df.columns:
            df[opt] = None
    if "neutral" not in df.columns:
        df["neutral"] = 0

    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = (
        df["neutral"].astype(str).str.lower().isin(["1", "true", "yes", "t"]).astype(int)
    )

    df = df.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])
    df = df[(df["home_team"] != "") & (df["away_team"] != "")]
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    cols = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ]
    return df[cols].sort_values("date").reset_index(drop=True)


def load_matches(con: sqlite3.Connection, csv_path) -> int:
    """Laedt historische Spiele aus einer CSV in die Tabelle ``matches``."""
    df = _coerce_matches(pd.read_csv(csv_path))
    df.to_sql("matches", con, if_exists="append", index=False)
    con.commit()
    return len(df)


def load_matches_df(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Wie ``load_matches``, aber direkt aus einem DataFrame."""
    df = _coerce_matches(df)
    df.to_sql("matches", con, if_exists="append", index=False)
    con.commit()
    return len(df)


def load_fixtures(con: sqlite3.Connection, csv_path) -> int:
    """Laedt anstehende Spiele (Fixtures) aus einer CSV."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for opt, default in (
        ("city", None), ("country", None), ("group_name", None),
        ("stage", "group"), ("neutral", 1),
    ):
        if opt not in df.columns:
            df[opt] = default
    if "match_id" not in df.columns:
        df["match_id"] = range(1, len(df) + 1)

    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)

    cols = [
        "match_id", "date", "home_team", "away_team", "city",
        "country", "neutral", "group_name", "stage",
    ]
    df = df[cols]
    df.to_sql("fixtures", con, if_exists="append", index=False)
    con.commit()
    return len(df)


def read_table(con: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Liest eine Tabelle vollstaendig in einen DataFrame."""
    if name not in {"matches", "fixtures"}:
        raise ValueError(f"Unbekannte Tabelle: {name}")
    return pd.read_sql_query(f"SELECT * FROM {name}", con)
