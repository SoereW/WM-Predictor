"""Laden offener Datenquellen.

Primaere Quelle fuer historische Laenderspiele:
``martj42/international_results`` (CC0, ~49k Spiele ab 1872). Das Schema
passt direkt auf unsere ``matches``-Tabelle.
"""

from __future__ import annotations

import time
from pathlib import Path

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def fetch_international_results(dest, url: str = RESULTS_URL, retries: int = 4, timeout: int = 60) -> Path:
    """Laedt die vollstaendige Ergebnis-Historie herunter und speichert sie.

    Mit einfachem Exponential-Backoff bei Netzwerkfehlern.
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return dest
        except Exception as err:  # noqa: BLE001 - bewusst breit, Netzwerk
            last_err = err
            wait = 2 ** (attempt + 1)
            print(f"Download fehlgeschlagen (Versuch {attempt + 1}/{retries}): {err} - warte {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Konnte historische Daten nicht laden: {last_err}")
