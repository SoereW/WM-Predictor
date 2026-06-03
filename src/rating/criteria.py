"""Kriterien, Positionsprofile und Gewichte - die zentrale Stellschraube.

Alle Bewertungen im Subsystem leiten sich aus *diesen* Definitionen ab.
Wer das Modell justieren will, aendert hier - nicht im Code der einzelnen
Bewertungsschritte. Gewichte sind bewusst explizit und summieren sich
(annaehernd) zu 1.0, damit ihr Einfluss direkt ablesbar ist.

Skala: Alle Attribute und Scores leben auf einer 0-100-Skala
(fussball-ueblich, vgl. EA/SoFIFA-Overall), damit Werte intuitiv sind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------
# Spieler-Attribute (die "genauen Kriterien", an denen jeder Spieler haengt).
# Roh-Input je Spieler; 0-100. Positionsprofile gewichten sie unterschiedlich.
# --------------------------------------------------------------------------
ATTRIBUTES: List[str] = [
    "pace",        # Tempo / Antritt
    "shooting",    # Abschluss / Torgefahr
    "passing",     # Passspiel / Spielaufbau
    "dribbling",   # Ballfuehrung / Technik
    "defending",   # Zweikampf / Stellungsspiel defensiv
    "physical",    # Physis / Ausdauer / Robustheit
    "vision",      # Spieluebersicht / Kreativitaet
    "gk",          # Torwart-Faehigkeit (nur fuer Keeper relevant)
]

# Positionslinien (grobe Gruppen). Detailpositionen werden hierauf gemappt.
LINES = ["GK", "DEF", "MID", "FWD"]

# Mapping Detailposition -> Linie + ein feineres "Rollenprofil".
_POSITION_MAP = {
    "GK": ("GK", "GK"), "TW": ("GK", "GK"), "GOALKEEPER": ("GK", "GK"),
    "CB": ("DEF", "CB"), "IV": ("DEF", "CB"), "RCB": ("DEF", "CB"), "LCB": ("DEF", "CB"),
    "RB": ("DEF", "FB"), "LB": ("DEF", "FB"), "RWB": ("DEF", "FB"), "LWB": ("DEF", "FB"),
    "DEF": ("DEF", "CB"),
    "CDM": ("MID", "DM"), "DM": ("MID", "DM"), "CM": ("MID", "CM"),
    "CAM": ("MID", "AM"), "AM": ("MID", "AM"), "LM": ("MID", "WM"), "RM": ("MID", "WM"),
    "MID": ("MID", "CM"),
    "LW": ("FWD", "W"), "RW": ("FWD", "W"), "W": ("FWD", "W"),
    "ST": ("FWD", "ST"), "CF": ("FWD", "ST"), "FW": ("FWD", "ST"), "FWD": ("FWD", "ST"),
}


def normalize_position(pos: object) -> str:
    """Detailposition -> grobe Linie (GK/DEF/MID/FWD)."""
    key = str(pos or "").strip().upper()
    return _POSITION_MAP.get(key, ("MID", "CM"))[0]


def position_role(pos: object) -> str:
    """Detailposition -> feineres Rollenkuerzel (CB/FB/DM/CM/AM/WM/W/ST/GK)."""
    key = str(pos or "").strip().upper()
    return _POSITION_MAP.get(key, ("MID", "CM"))[1]


def position_line(pos: object) -> str:
    """Alias fuer normalize_position (sprechender Name an Aufrufstellen)."""
    return normalize_position(pos)


# --------------------------------------------------------------------------
# Positionsprofile: wie stark zaehlt welches Attribut fuer welche Rolle?
# Jede Zeile summiert sich zu 1.0. Das ist die fussballerische Kernlogik:
# ein Stuermer wird an Abschluss/Tempo gemessen, ein IV an Defending/Physis.
# --------------------------------------------------------------------------
ROLE_PROFILES: Dict[str, Dict[str, float]] = {
    "GK": {"gk": 0.85, "physical": 0.08, "passing": 0.07},
    "CB": {"defending": 0.42, "physical": 0.24, "passing": 0.14, "pace": 0.12, "vision": 0.08},
    "FB": {"defending": 0.26, "pace": 0.24, "physical": 0.16, "passing": 0.16, "dribbling": 0.10, "vision": 0.08},
    "DM": {"defending": 0.34, "passing": 0.22, "physical": 0.18, "vision": 0.16, "dribbling": 0.10},
    "CM": {"passing": 0.26, "vision": 0.22, "dribbling": 0.16, "defending": 0.16, "physical": 0.12, "shooting": 0.08},
    "AM": {"vision": 0.24, "passing": 0.22, "dribbling": 0.22, "shooting": 0.18, "pace": 0.14},
    "WM": {"pace": 0.24, "passing": 0.20, "dribbling": 0.22, "vision": 0.16, "defending": 0.10, "shooting": 0.08},
    "W":  {"pace": 0.26, "dribbling": 0.26, "shooting": 0.20, "passing": 0.14, "vision": 0.14},
    "ST": {"shooting": 0.40, "pace": 0.22, "physical": 0.16, "dribbling": 0.12, "vision": 0.10},
}


@dataclass(frozen=True)
class PlayerWeights:
    """Gewichte fuer den Spieler-Gesamtscore.

    Setzt sich zusammen aus dem positionsspezifischen *Skill* (aus den
    Attributen), der aktuellen *Form*, der *Erfahrung* (Laenderspiele/Alter)
    und der *Verfuegbarkeit/Fitness*. Summe ~ 1.0.
    """
    skill: float = 0.68
    form: float = 0.16
    experience: float = 0.10
    fitness: float = 0.06

    def as_dict(self) -> Dict[str, float]:
        return {"skill": self.skill, "form": self.form, "experience": self.experience, "fitness": self.fitness}


@dataclass(frozen=True)
class TeamWeights:
    """Gewichte fuer den Team-Gesamtscore.

    Beste Elf (Hauptfaktor), Kadertiefe (Reservequalitaet), Chemie
    (Zusammenspiel) und Trainer/Coaching. Summe ~ 1.0.
    """
    best_xi: float = 0.62
    depth: float = 0.13
    chemistry: float = 0.17
    coach: float = 0.08

    def as_dict(self) -> Dict[str, float]:
        return {"best_xi": self.best_xi, "depth": self.depth, "chemistry": self.chemistry, "coach": self.coach}


@dataclass(frozen=True)
class ChemistryWeights:
    """Gewichte der Chemie-Teilkriterien. Summe ~ 1.0.

    - club_blocks   : Spieler aus denselben Vereinen (eingespielte Bloecke)
    - cohesion      : Eingespieltheit ueber gemeinsame Laenderspiele/Verweildauer
    - positional    : spielen die Spieler auf ihrer Stammposition?
    - age_balance   : ausgewogene Altersstruktur (nicht zu alt/zu jung)
    - coach_stability: Kontinuitaet auf der Trainerbank
    """
    club_blocks: float = 0.30
    cohesion: float = 0.28
    positional: float = 0.18
    age_balance: float = 0.12
    coach_stability: float = 0.12

    def as_dict(self) -> Dict[str, float]:
        return {
            "club_blocks": self.club_blocks,
            "cohesion": self.cohesion,
            "positional": self.positional,
            "age_balance": self.age_balance,
            "coach_stability": self.coach_stability,
        }


# Ziel-Formation fuer die beste Elf (Anzahl je Linie).
FORMATION: Dict[str, int] = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}

# Positions-Wichtigkeit der besten Elf ("wie es in echt ist"): die Achse
# (Torwart, Innenverteidigung, Sechser, Mittelstuermer) entscheidet Spiele
# staerker als die aeusseren Positionen. Gewichte sind bewusst mild (um 1.0),
# damit sie die Rangfolge nur fein justieren. Validiert: senkt den
# Out-of-Sample-Log-Loss der Kader-Vorhersage leicht und konsistent.
ROLE_IMPORTANCE: Dict[str, float] = {
    "GK": 1.25, "CB": 1.15, "FB": 0.85, "DM": 1.15, "CM": 1.05,
    "AM": 1.05, "WM": 0.90, "W": 0.95, "ST": 1.15,
}

# Star-Gewichtung: zusaetzlich werden die staerksten Spieler der Elf etwas
# hoeher gewichtet (Spitzenqualitaet entscheidet, besonders im K.o.). 0 =
# reiner positionsgewichteter Mittelwert, 1 = rein an Spitzenstaerke. Mild.
STAR_ALPHA: float = 0.25

# Neutrale Chemie: ab diesem Wert wirkt Chemie weder hebend noch daempfend.
# Chemie wird als Abweichung hiervon (gekoppelt ans Niveau) verrechnet,
# damit perfekte Chemie eines schwachen Kaders ihn nicht kuenstlich hochzieht.
NEUTRAL_CHEMISTRY = 75.0

# Idealer Altersbereich (Peak) fuer die Altersbalance-Bewertung.
AGE_PEAK_LOW = 24
AGE_PEAK_HIGH = 30

# Liga-Staerke 0-1 (grob). Ein eingespielter Vereinsblock aus einer
# Top-Liga ist wertvoller als aus einer schwaecheren - das fliesst in die
# club_blocks-Chemie ein. Unbekannte Ligen bekommen einen mittleren Wert.
LEAGUE_STRENGTH: Dict[str, float] = {
    # Top-5 Europa
    "Premier League": 1.00, "La Liga": 0.96, "Serie A": 0.92,
    "Bundesliga": 0.92, "Ligue 1": 0.86,
    # weitere europaeische
    "Primeira Liga": 0.78, "Eredivisie": 0.76, "Liga Portugal": 0.78,
    "Belgian Pro League": 0.70, "Super Lig": 0.68, "Championship": 0.66,
    # ausserhalb / Top-Clubs
    "MLS": 0.55, "Saudi Pro League": 0.62, "Liga MX": 0.58,
    "J1 League": 0.55, "Brasileirao": 0.66, "Liga Profesional": 0.62,
}
DEFAULT_LEAGUE_STRENGTH = 0.55

# Zuordnung bekannter Vereine -> Liga (nur fuer die Stichprobe noetig; bei
# echten Daten kann eine Spalte ``league`` direkt mitgeliefert werden).
CLUB_LEAGUE: Dict[str, str] = {
    "Aston Villa": "Premier League", "Tottenham": "Premier League",
    "Liverpool": "Premier League", "Chelsea": "Premier League",
    "Arsenal": "Premier League", "Man City": "Premier League",
    "Newcastle": "Premier League", "West Ham": "Premier League",
    "Brighton": "Premier League", "Crystal Palace": "Premier League",
    "Barcelona": "La Liga", "Real Madrid": "La Liga",
    "Atletico Madrid": "La Liga", "Real Sociedad": "La Liga",
    "Inter": "Serie A", "AC Milan": "Serie A", "Juventus": "Serie A",
    "Parma": "Serie A",
    "Bayern": "Bundesliga", "Leverkusen": "Bundesliga", "Leipzig": "Bundesliga",
    "Monchengladbach": "Bundesliga",
    "PSG": "Ligue 1", "Marseille": "Ligue 1", "Lyon": "Ligue 1", "Monaco": "Ligue 1",
    "Benfica": "Primeira Liga", "Porto": "Primeira Liga", "Sporting": "Primeira Liga",
    "Feyenoord": "Eredivisie",
    "Inter Miami": "MLS",
    "Al-Nassr": "Saudi Pro League", "Al-Hilal": "Saudi Pro League",
    "Al-Ahli": "Saudi Pro League",
}


def league_strength(club: object, league: object = None) -> float:
    """Liga-Staerke 0-1 fuer einen Verein (oder direkt eine Liga)."""
    if league is not None and str(league).strip():
        return LEAGUE_STRENGTH.get(str(league).strip(), DEFAULT_LEAGUE_STRENGTH)
    lg = CLUB_LEAGUE.get(str(club).strip()) if club is not None else None
    return LEAGUE_STRENGTH.get(lg, DEFAULT_LEAGUE_STRENGTH) if lg else DEFAULT_LEAGUE_STRENGTH
