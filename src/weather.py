"""Optionale Wetter-Integration (Open-Meteo).

Erweiterungspunkt: liefert Temperatur/Niederschlag pro Ort und Datum und
uebersetzt sie in ein kleines Stil-Adjustment. Bewusst robust gehalten -
ohne Netzwerk oder bei Fehlern wird ``None`` bzw. ``0.0`` zurueckgegeben,
damit nie etwas haengen bleibt. Im Dashboard wird das Wetter aktuell
manuell ueber den ``weather_delta``-Slider gesteuert; dieses Modul macht
die Automatisierung anschliessbar.
"""

from __future__ import annotations

from typing import Dict, Optional

# Grobe Koordinaten einiger WM-2026-Spielorte (Host: USA/Kanada/Mexiko).
HOST_CITY_COORDS: Dict[str, tuple] = {
    "New York": (40.71, -74.01),
    "Los Angeles": (34.05, -118.24),
    "Dallas": (32.78, -96.80),
    "Miami": (25.76, -80.19),
    "Atlanta": (33.75, -84.39),
    "Seattle": (47.61, -122.33),
    "Houston": (29.76, -95.37),
    "Kansas City": (39.10, -94.58),
    "Boston": (42.36, -71.06),
    "Philadelphia": (39.95, -75.17),
    "San Francisco": (37.77, -122.42),
    "Toronto": (43.65, -79.38),
    "Vancouver": (49.28, -123.12),
    "Mexico City": (19.43, -99.13),
    "Guadalajara": (20.67, -103.35),
    "Monterrey": (25.69, -100.32),
}


def _first(seq):
    return seq[0] if seq else None


def fetch_weather(lat: float, lon: float, date: str, timeout: int = 15) -> Optional[Dict]:
    """Holt Tageswetter von Open-Meteo. Gibt ``None`` bei Fehlern zurueck."""
    try:
        import requests

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max",
            "timezone": "UTC",
        }
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        return {
            "temp_max": _first(daily.get("temperature_2m_max")),
            "precip": _first(daily.get("precipitation_sum")),
            "wind": _first(daily.get("wind_speed_10m_max")),
        }
    except Exception:
        return None


def weather_to_adjustment(weather: Optional[Dict]) -> float:
    """Heuristik: extremes Wetter daempft das technisch staerkere Team leicht.

    Rueckgabe im Bereich ~[-0.15, 0.0] als Daempfung; 0.0 wenn unbekannt.
    Bewusst konservativ - Wetter ist ein schwaches Signal.
    """
    if not weather:
        return 0.0
    penalty = 0.0
    temp = weather.get("temp_max")
    precip = weather.get("precip")
    wind = weather.get("wind")
    if temp is not None and temp >= 32:
        penalty -= 0.05
    if precip is not None and precip >= 10:
        penalty -= 0.05
    if wind is not None and wind >= 40:
        penalty -= 0.05
    return round(max(penalty, -0.15), 3)
