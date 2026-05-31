"""Kuratierte Naeherungsdaten bekannter Spieler - fuer Stichproben-Tests.

ZWECK: Das Bewertungssystem laesst sich nicht an der Vergangenheit
backtesten (es gibt keine "wahre" Chemie-Zahl). Stattdessen pruefen wir es
an *bekannten* Spielern und Teams: Stimmen die relativen Einordnungen mit
der fussballerischen Realitaet ueberein (Top-Stuermer > Rollenspieler,
Top-Nation > Aussenseiter)?

WICHTIG: Alle Attribute sind **Naeherungen/Schaetzungen** (Groessenordnung
EA/SoFIFA), keine offiziellen Werte. Spielerstaerken, Caps und Alter sind
auf den Stand grob 2024/25 gesetzt und dienen ausschliesslich dem Test der
*Bewertungslogik*, nicht als Wahrheit ueber einzelne Personen.

Format pro Spieler: name, team, club, position, natural_position, age,
caps, form, available + Attribute (pace..gk).
"""

from __future__ import annotations

import pandas as pd

# (name, team, club, pos, age, caps, form, pace, sho, pas, dri, def, phy, vis, gk)
_P = [
    # ---- Argentina ----
    ("Emiliano Martinez", "Argentina", "Aston Villa", "GK", 32, 42, 82, 50, 30, 70, 45, 30, 82, 60, 89),
    ("Nahuel Molina", "Argentina", "Atletico Madrid", "RB", 26, 40, 75, 84, 55, 74, 76, 76, 80, 72, 10),
    ("Cristian Romero", "Argentina", "Tottenham", "CB", 26, 38, 84, 78, 45, 74, 70, 88, 86, 72, 10),
    ("Nicolas Otamendi", "Argentina", "Benfica", "CB", 36, 120, 78, 68, 48, 72, 60, 85, 84, 70, 10),
    ("Nicolas Tagliafico", "Argentina", "Lyon", "LB", 32, 60, 76, 82, 50, 75, 74, 78, 82, 74, 10),
    ("Rodrigo De Paul", "Argentina", "Atletico Madrid", "CM", 30, 70, 80, 80, 70, 84, 82, 78, 84, 86, 10),
    ("Enzo Fernandez", "Argentina", "Chelsea", "CM", 24, 35, 82, 76, 76, 88, 82, 70, 80, 90, 10),
    ("Alexis Mac Allister", "Argentina", "Liverpool", "CM", 26, 40, 85, 74, 80, 87, 84, 74, 82, 89, 10),
    ("Lionel Messi", "Argentina", "Inter Miami", "RW", 37, 190, 90, 80, 92, 91, 95, 38, 68, 96, 10),
    ("Julian Alvarez", "Argentina", "Atletico Madrid", "ST", 24, 40, 86, 86, 86, 80, 86, 70, 82, 84, 10),
    ("Lautaro Martinez", "Argentina", "Inter", "ST", 27, 65, 84, 86, 89, 76, 86, 74, 84, 80, 10),
    ("Angel Di Maria", "Argentina", "Benfica", "RW", 36, 145, 80, 82, 82, 86, 90, 45, 66, 90, 10),
    # ---- France ----
    ("Mike Maignan", "France", "AC Milan", "GK", 29, 30, 84, 55, 30, 76, 50, 35, 80, 64, 88),
    ("Jules Kounde", "France", "Barcelona", "RB", 26, 40, 82, 86, 45, 80, 78, 84, 86, 74, 10),
    ("Dayot Upamecano", "France", "Bayern", "CB", 26, 30, 80, 82, 40, 74, 66, 86, 88, 70, 10),
    ("William Saliba", "France", "Arsenal", "CB", 23, 20, 85, 82, 42, 78, 72, 88, 87, 74, 10),
    ("Theo Hernandez", "France", "AC Milan", "LB", 27, 35, 82, 90, 62, 78, 82, 78, 86, 78, 10),
    ("Aurelien Tchouameni", "France", "Real Madrid", "DM", 24, 35, 84, 74, 60, 82, 78, 84, 86, 80, 10),
    ("Adrien Rabiot", "France", "Marseille", "CM", 29, 48, 80, 76, 64, 82, 80, 78, 84, 80, 10),
    ("Antoine Griezmann", "France", "Atletico Madrid", "AM", 33, 130, 84, 78, 84, 86, 90, 60, 78, 92, 10),
    ("Kylian Mbappe", "France", "Real Madrid", "ST", 26, 80, 90, 97, 90, 80, 92, 40, 78, 88, 10),
    ("Ousmane Dembele", "France", "PSG", "RW", 27, 50, 83, 92, 80, 84, 92, 45, 72, 84, 10),
    ("Marcus Thuram", "France", "Inter", "ST", 27, 25, 82, 86, 82, 74, 82, 82, 84, 76, 10),
    ("Randal Kolo Muani", "France", "PSG", "ST", 26, 30, 78, 88, 80, 76, 80, 78, 82, 78, 10),
    # ---- Brazil ----
    ("Alisson", "Brazil", "Liverpool", "GK", 32, 70, 84, 52, 30, 78, 48, 35, 84, 66, 89),
    ("Danilo", "Brazil", "Juventus", "RB", 33, 55, 76, 76, 55, 78, 74, 80, 82, 76, 10),
    ("Marquinhos", "Brazil", "PSG", "CB", 30, 90, 84, 78, 45, 80, 68, 86, 84, 78, 10),
    ("Gabriel Magalhaes", "Brazil", "Arsenal", "CB", 27, 25, 84, 78, 48, 74, 64, 88, 88, 72, 10),
    ("Wendell", "Brazil", "Porto", "LB", 31, 12, 74, 80, 52, 76, 72, 76, 80, 74, 10),
    ("Bruno Guimaraes", "Brazil", "Newcastle", "DM", 27, 38, 84, 74, 64, 86, 82, 82, 84, 86, 10),
    ("Lucas Paqueta", "Brazil", "West Ham", "AM", 27, 50, 78, 80, 72, 86, 88, 70, 80, 88, 10),
    ("Rodrygo", "Brazil", "Real Madrid", "RW", 24, 35, 84, 90, 82, 82, 90, 50, 72, 86, 10),
    ("Vinicius Junior", "Brazil", "Real Madrid", "LW", 24, 40, 88, 95, 80, 78, 94, 45, 70, 86, 10),
    ("Raphinha", "Brazil", "Barcelona", "RW", 28, 40, 86, 88, 84, 84, 88, 58, 76, 88, 10),
    ("Endrick", "Brazil", "Real Madrid", "ST", 18, 12, 76, 86, 82, 70, 80, 72, 78, 72, 10),
    # ---- Germany ----
    ("Marc-Andre ter Stegen", "Germany", "Barcelona", "GK", 32, 45, 82, 55, 30, 84, 52, 35, 82, 70, 88),
    ("Joshua Kimmich", "Germany", "Bayern", "RB", 29, 95, 84, 76, 70, 88, 84, 78, 84, 90, 10),
    ("Antonio Ruediger", "Germany", "Real Madrid", "CB", 31, 75, 84, 82, 44, 76, 66, 88, 88, 72, 10),
    ("Jonathan Tah", "Germany", "Leverkusen", "CB", 28, 30, 82, 80, 42, 74, 64, 86, 88, 70, 10),
    ("David Raum", "Germany", "Leipzig", "LB", 26, 25, 78, 82, 55, 78, 78, 74, 80, 80, 10),
    ("Toni Kroos", "Germany", "Real Madrid", "CM", 34, 114, 86, 60, 90, 90, 92, 64, 78, 94, 10),
    ("Ilkay Gundogan", "Germany", "Man City", "CM", 33, 80, 82, 68, 78, 86, 86, 70, 80, 90, 10),
    ("Jamal Musiala", "Germany", "Bayern", "AM", 21, 35, 88, 88, 82, 82, 94, 55, 74, 88, 10),
    ("Florian Wirtz", "Germany", "Leverkusen", "AM", 21, 25, 88, 84, 84, 88, 92, 55, 72, 92, 10),
    ("Kai Havertz", "Germany", "Arsenal", "ST", 25, 50, 82, 80, 82, 80, 84, 70, 82, 84, 10),
    ("Leroy Sane", "Germany", "Bayern", "RW", 28, 65, 82, 90, 80, 82, 88, 50, 74, 84, 10),
    # ---- Japan (starke Elo-Historie, Test fuer Kader vs. Historie) ----
    ("Zion Suzuki", "Japan", "Parma", "GK", 22, 12, 76, 55, 30, 70, 48, 35, 78, 60, 82),
    ("Hiroki Ito", "Japan", "Bayern", "CB", 25, 25, 78, 76, 45, 76, 64, 80, 82, 72, 10),
    ("Ko Itakura", "Japan", "Monchengladbach", "CB", 27, 35, 78, 74, 44, 74, 62, 80, 82, 70, 10),
    ("Takehiro Tomiyasu", "Japan", "Arsenal", "RB", 26, 40, 76, 78, 48, 76, 70, 82, 82, 72, 10),
    ("Wataru Endo", "Japan", "Liverpool", "DM", 31, 60, 78, 72, 55, 78, 74, 82, 84, 76, 10),
    ("Hidemasa Morita", "Japan", "Sporting", "DM", 29, 45, 76, 74, 60, 80, 76, 76, 82, 80, 10),
    ("Takefusa Kubo", "Japan", "Real Sociedad", "RW", 23, 35, 84, 86, 76, 88, 88, 55, 72, 86, 10),
    ("Kaoru Mitoma", "Japan", "Brighton", "LW", 27, 35, 84, 86, 74, 88, 90, 58, 76, 84, 10),
    ("Daichi Kamada", "Japan", "Crystal Palace", "AM", 28, 40, 78, 76, 76, 82, 82, 64, 78, 86, 10),
    ("Takumi Minamino", "Japan", "Monaco", "AM", 29, 55, 78, 80, 74, 82, 82, 60, 78, 82, 10),
    ("Ayase Ueda", "Japan", "Feyenoord", "ST", 26, 25, 76, 82, 80, 72, 76, 76, 80, 72, 10),
    # ---- Saudi Arabia (Aussenseiter-Referenz) ----
    ("Nawaf Al-Aqidi", "Saudi Arabia", "Al-Nassr", "GK", 24, 10, 68, 50, 28, 60, 42, 30, 70, 52, 72),
    ("Sultan Al-Ghanam", "Saudi Arabia", "Al-Nassr", "RB", 30, 40, 68, 74, 45, 66, 62, 70, 72, 64, 10),
    ("Ali Al-Bulaihi", "Saudi Arabia", "Al-Hilal", "CB", 35, 55, 70, 66, 40, 64, 55, 76, 78, 62, 10),
    ("Hassan Tambakti", "Saudi Arabia", "Al-Hilal", "CB", 25, 25, 70, 72, 42, 66, 58, 74, 76, 64, 10),
    ("Yasir Al-Shahrani", "Saudi Arabia", "Al-Hilal", "LB", 32, 70, 70, 76, 48, 70, 66, 70, 74, 68, 10),
    ("Mohamed Kanno", "Saudi Arabia", "Al-Hilal", "DM", 30, 55, 70, 70, 52, 70, 66, 72, 76, 70, 10),
    ("Salem Al-Dawsari", "Saudi Arabia", "Al-Hilal", "LW", 33, 75, 76, 80, 76, 80, 82, 55, 70, 80, 10),
    ("Firas Al-Buraikan", "Saudi Arabia", "Al-Ahli", "ST", 24, 35, 72, 80, 74, 68, 72, 72, 76, 68, 10),
    ("Abdullah Al-Hamdan", "Saudi Arabia", "Al-Hilal", "ST", 25, 30, 70, 78, 72, 66, 70, 70, 74, 66, 10),
    ("Nasser Al-Dawsari", "Saudi Arabia", "Al-Hilal", "CM", 26, 30, 70, 72, 64, 74, 72, 68, 74, 74, 10),
    ("Sami Al-Najei", "Saudi Arabia", "Al-Nassr", "CM", 27, 35, 70, 74, 62, 74, 72, 66, 74, 74, 10),
]

# Trainer-Naeherungen (tenure_years grob, major_tournaments = Endrunden).
COACHES = {
    "Argentina": {"name": "Lionel Scaloni", "tenure_years": 6, "major_tournaments": 3, "rating": 88},
    "France": {"name": "Didier Deschamps", "tenure_years": 12, "major_tournaments": 6, "rating": 86},
    "Brazil": {"name": "Dorival Junior", "tenure_years": 1, "major_tournaments": 0, "rating": 74},
    "Germany": {"name": "Julian Nagelsmann", "tenure_years": 2, "major_tournaments": 1, "rating": 80},
    "Japan": {"name": "Hajime Moriyasu", "tenure_years": 6, "major_tournaments": 2, "rating": 78},
    "Saudi Arabia": {"name": "Herve Renard", "tenure_years": 1, "major_tournaments": 1, "rating": 75},
}

_COLS = [
    "name", "team", "club", "natural_position", "age", "caps", "form",
    "pace", "shooting", "passing", "dribbling", "defending", "physical", "vision", "gk",
]


def load_sample_players() -> pd.DataFrame:
    """Liefert die kuratierten Beispielspieler als DataFrame im Modul-Format."""
    df = pd.DataFrame(_P, columns=_COLS)
    df["player"] = df["name"]
    df["position"] = df["natural_position"]
    df["available"] = 1
    return df
