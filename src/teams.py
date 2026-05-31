"""Normalisierung von Teamnamen.

Historische Datensaetze (z. B. martj42/international_results) und
FIFA-Spielplaene benutzen teils unterschiedliche Schreibweisen fuer
dasselbe Team. Ohne Normalisierung zerfaellt die Historie eines Teams in
mehrere "Phantom-Teams" und Elo/Form werden unbrauchbar. Diese Tabelle
fuehrt die wichtigsten Varianten auf einen kanonischen Namen zusammen.
"""

from __future__ import annotations

# Variante -> kanonischer Name. Schluessel werden case-insensitiv und ohne
# fuehrende/abschliessende Leerzeichen verglichen.
_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea dpr": "North Korea",
    "dpr korea": "North Korea",
    "ir iran": "Iran",
    "iran (islamic republic of)": "Iran",
    "china pr": "China",
    "chinese taipei": "Taiwan",
    "czechia": "Czech Republic",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "cabo verde": "Cape Verde",
    "cape verde islands": "Cape Verde",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "congo-kinshasa": "DR Congo",
    "drc": "DR Congo",
    "republic of the congo": "Congo",
    "congo-brazzaville": "Congo",
    "bosnia and herzegovina": "Bosnia-Herzegovina",
    "bosnia": "Bosnia-Herzegovina",
    "north macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "fyr macedonia": "North Macedonia",
    "the gambia": "Gambia",
    "kyrgyz republic": "Kyrgyzstan",
    "brunei darussalam": "Brunei",
    "st kitts and nevis": "Saint Kitts and Nevis",
    "st lucia": "Saint Lucia",
    "st vincent and the grenadines": "Saint Vincent and the Grenadines",
    "curaçao": "Curacao",
    "hong kong china": "Hong Kong",
    "republic of ireland": "Ireland",
    "great britain": "United Kingdom",
    "swaziland": "Eswatini",
    "uae": "United Arab Emirates",
    "viet nam": "Vietnam",
}


def normalize_team(name: object) -> str:
    """Fuehrt eine Schreibweise auf den kanonischen Teamnamen zurueck."""
    if name is None:
        return ""
    text = str(name).strip()
    if not text:
        return ""
    key = text.lower()
    return _ALIASES.get(key, text)


def normalize_series(series):
    """Vektorisierte Normalisierung fuer eine pandas-Series."""
    return series.map(normalize_team)
