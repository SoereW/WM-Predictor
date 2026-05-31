"""Spielkontext-Faktoren fuer den Vorhersagezeitpunkt.

Diese Faktoren lassen sich nicht sauber an der Historie backtesten (sie
brauchen Reisewege, Anstosszeiten, Hoehenlagen etc., die im Ergebnis-
Datensatz fehlen). Sie werden daher als **gewichtete, beschraenkte
Elo-Korrekturen** modelliert und greifen - wie die Kaderstaerke - erst zur
Vorhersage. Jeder Teilfaktor ist gedeckelt, damit kein einzelner Aspekt die
Prognose dominiert; die Summe wird zusaetzlich begrenzt.

Modellierte Faktoren (jeweils aus Sicht des Heimteams, Punkte auf Elo-Skala):
- **Ruhe/Pause**: Differenz der Ruhetage seit dem letzten Spiel (frischere
  Beine sind ein leichter Vorteil; saettigt schnell).
- **Reise**: zurueckgelegte Distanz zum Spielort + Zeitzonen/Jetlag.
- **Hoehe**: Anpassung an die Stadionhoehe (Heimvorteil fuer
  hoehengewohnte Teams, Nachteil fuer Flachland-Teams in der Hoehe).
- **Klima**: Hitze/Luftfeuchte relativ zur Heimatklimazone.

Alle Werte sind bewusst konservativ - Kontext ist ein Feinschliff, kein
Hauptsignal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Obergrenzen je Teilfaktor (Elo-Punkte) und fuer die Summe.
CAP_REST = 25.0
CAP_TRAVEL = 30.0
CAP_ALTITUDE = 40.0
CAP_CLIMATE = 25.0
CAP_TOTAL = 70.0

EARTH_KM = 6371.0


@dataclass
class ContextResult:
    elo_delta: float                       # Gesamt-Korrektur (Heimsicht), Elo
    parts: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


def _haversine(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Grosskreis-Distanz zwischen zwei (lat, lon) in km."""
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def _clip(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


def rest_delta(home_rest_days: Optional[float], away_rest_days: Optional[float]) -> float:
    """Vorteil aus mehr Ruhetagen (saettigend, gedeckelt)."""
    if home_rest_days is None or away_rest_days is None:
        return 0.0
    # Nutzen pro Ruhetag nimmt ab; ab ~6 Tagen kaum noch Unterschied.
    def util(d: float) -> float:
        return 1.0 - math.exp(-max(d, 0) / 3.5)
    diff = util(home_rest_days) - util(away_rest_days)
    return _clip(diff * CAP_REST * 1.4, CAP_REST)


def travel_delta(
    home_coord: Optional[Tuple[float, float]],
    away_coord: Optional[Tuple[float, float]],
    venue_coord: Optional[Tuple[float, float]],
    home_tz: Optional[float] = None,
    away_tz: Optional[float] = None,
    venue_tz: Optional[float] = None,
) -> float:
    """Reisenachteil: weiter gereistes Team ist leicht im Nachteil.

    Beruecksichtigt Distanz zum Spielort und (falls bekannt) Zeitzonen-
    Verschiebung (Jetlag).
    """
    if not (home_coord and away_coord and venue_coord):
        return 0.0
    d_home = _haversine(home_coord, venue_coord)
    d_away = _haversine(away_coord, venue_coord)
    # Pro 1000 km ~6 Elo, saettigend ueber Wurzel (Langstrecke zaehlt weniger linear).
    def fatigue(km: float) -> float:
        return math.sqrt(max(km, 0) / 1000.0) * 9.0
    delta = fatigue(d_away) - fatigue(d_home)  # Heimvorteil, wenn Gast weiter reist

    if home_tz is not None and away_tz is not None and venue_tz is not None:
        jet = (abs(away_tz - venue_tz) - abs(home_tz - venue_tz)) * 3.0
        delta += jet
    return _clip(delta, CAP_TRAVEL)


def altitude_delta(
    venue_alt_m: Optional[float],
    home_alt_m: Optional[float],
    away_alt_m: Optional[float],
) -> float:
    """Hoehen-Effekt: an Hoehe gewoehnte Teams im Vorteil, wenn das Spiel
    hoch liegt (und umgekehrt fuer Flachland-Teams)."""
    if venue_alt_m is None:
        return 0.0
    home_alt = home_alt_m if home_alt_m is not None else 100.0
    away_alt = away_alt_m if away_alt_m is not None else 100.0
    # Relevanz erst ab ~1500 m; darunter vernachlaessigbar.
    if venue_alt_m < 1200:
        return 0.0
    severity = min((venue_alt_m - 1200) / 2500.0, 1.0)  # 0..1 bis ~3700 m
    # Wer naeher an der Spielhoehe lebt, ist besser akklimatisiert.
    home_gap = abs(venue_alt_m - home_alt)
    away_gap = abs(venue_alt_m - away_alt)
    # Differenz der Akklimatisierungs-Luecke (kleiner = besser).
    delta = (away_gap - home_gap) / 2500.0 * CAP_ALTITUDE * severity
    return _clip(delta, CAP_ALTITUDE)


def climate_delta(
    venue_temp_c: Optional[float],
    venue_humidity: Optional[float],
    home_climate: Optional[str] = None,
    away_climate: Optional[str] = None,
) -> float:
    """Klima-Effekt: Hitze/Schwuele beguenstigt das daran gewoehnte Team."""
    if venue_temp_c is None:
        return 0.0
    # Hitze-Index grob: nur hohe Temperaturen sind relevant.
    if venue_temp_c < 26:
        return 0.0
    heat = min((venue_temp_c - 26) / 12.0, 1.0)  # 0..1 bis ~38 C
    if venue_humidity is not None:
        heat = min(heat * (0.7 + 0.6 * min(venue_humidity / 100.0, 1.0)), 1.0)

    warm = {"tropical", "hot", "arid", "warm", "subtropical"}
    home_warm = (home_climate or "").lower() in warm
    away_warm = (away_climate or "").lower() in warm
    if home_warm == away_warm:
        return 0.0
    sign = 1.0 if home_warm else -1.0  # warmes Team profitiert bei Hitze
    return _clip(sign * heat * CAP_CLIMATE, CAP_CLIMATE)


def compute_context(
    home: Dict,
    away: Dict,
    venue: Dict,
) -> ContextResult:
    """Aggregiert alle Kontext-Teilfaktoren zu einer Elo-Korrektur (Heimsicht).

    ``home``/``away`` koennen enthalten: ``rest_days``, ``coord`` (lat,lon),
    ``tz`` (Stunden), ``altitude_m``, ``climate``. ``venue``: ``coord``,
    ``tz``, ``altitude_m``, ``temp_c``, ``humidity``. Fehlende Angaben
    deaktivieren den jeweiligen Teilfaktor (Wert 0).
    """
    parts = {
        "rest": round(rest_delta(home.get("rest_days"), away.get("rest_days")), 1),
        "travel": round(
            travel_delta(
                home.get("coord"), away.get("coord"), venue.get("coord"),
                home.get("tz"), away.get("tz"), venue.get("tz"),
            ),
            1,
        ),
        "altitude": round(
            altitude_delta(venue.get("altitude_m"), home.get("altitude_m"), away.get("altitude_m")),
            1,
        ),
        "climate": round(
            climate_delta(
                venue.get("temp_c"), venue.get("humidity"),
                home.get("climate"), away.get("climate"),
            ),
            1,
        ),
    }
    total = _clip(sum(parts.values()), CAP_TOTAL)
    return ContextResult(elo_delta=round(total, 1), parts=parts)
