# Daten- und Modell-Audit – WM 2026 Predictor

Stand der Überarbeitung: 2026-06-10

## Kurzfazit

Das Projekt war grundsätzlich lauffähig, aber die wichtigste Datenbasis für die WM-Prognose war nicht gut genug: Der Gruppenspielplan war im Upload nicht der offizielle FIFA-Spielplan, sondern ein vereinfachter Generator mit falschen Datums-/Venue-Kombinationen. Außerdem enthielt die Historien-CSV bereits zukünftige WM-Fixtures ohne Endergebnis. Beides wurde korrigiert.

Der Algorithmus ist kein „bestmöglicher“ Algorithmus im absoluten Sinn. Für dieses Datenproblem ist der gewählte hybride Ansatz aber sinnvoller als ein unnötig komplexes Deep-Learning-Modell: Internationale Fußballspiele sind relativ wenige, ungleich verteilt und stark von Kader-/Kontextinformationen abhängig. Deshalb bleibt das Projekt bei einem nachvollziehbaren Elo/Form/Poisson/Logit-Ensemble und verbessert die Daten- und Turnierlogik.

## Geänderte Daten

### 1. Offizieller Gruppenspielplan

- `src/wm2026.py` enthält jetzt `OFFICIAL_GROUP_FIXTURES` mit allen 72 Gruppenspielen.
- `data/wm2026_fixtures.csv` wurde daraus neu erzeugt.
- Je Fixture sind Datum, Heim-/Auswärtsteam, Stadt, Land, Neutralitätsflag und Gruppe enthalten.
- Venue-Koordinaten wurden ergänzt, damit Reise-/Kontextfeatures spielgenau berechnet werden.

Beispielhafte Korrektur:

- Vorher: gruppenbasierte, generierte Paarungen/Termine; dadurch konnte z. B. ein Spiel am falschen Datum oder im falschen Stadion landen.
- Jetzt: fixer Spielplan laut FIFA-Fixture-Liste, z. B. Mexiko – Südafrika am 2026-06-11 in Mexico City.

### 2. Historische Ergebnisse bereinigt

- `data/international_results.csv` enthält jetzt nur abgeschlossene Spiele mit Ergebnis.
- Entfernt wurden zukünftige WM-Fixtures ohne Score (`NaN`).
- Ergebnis nach Bereinigung: 49.339 abgeschlossene Länderspiele.
- Letzter Eintrag in der mitgelieferten Historie: 2026-06-05.

Warum das wichtig ist: Future-Fixtures ohne Ergebnis gehören nicht in die Trainings-/Elo-Historie. Auch wenn sie nicht direkt als Ergebnis trainiert wurden, können solche Zeilen später bei Datenbank- oder Feature-Logik stille Fehler verursachen.

### 3. Kader-/Spielerdaten eingeordnet

Die vorhandenen Kaderdaten sind nutzbar, aber nicht perfekt:

- `data/wm2026_squads.csv` ist eine Näherung über verfügbare Spieler-/Ratingdaten.
- Es sind keine offiziell nominierten WM-Kader, weil diese zum Prüfzeitpunkt noch nicht feststehen.
- Einige Nationen haben dünnere Spielerabdeckung (`n_players` in `wm2026_team_ratings.csv`), insbesondere Teams mit weniger gut abgedeckten Ligen.
- `data/fc26_players.csv` ist nicht im Projekt enthalten. Die Build-Pipeline kann deshalb offline nur mit der bereits mitgelieferten `wm2026_squads.csv` reproduzieren.

Verbesserung: `scripts/build_wm2026.py` nutzt jetzt automatisch die vorhandene `wm2026_squads.csv`, falls der große Rohdatensatz fehlt. Mit `--refresh` kann ein Online-Rebuild bewusst angestoßen werden.

## Geänderte Modell-/Turnierlogik

### 1. Spielgenauer Kontext

Vorher wurde der Reise-/Kontextvorteil vereinfacht über Gruppen-/Gastgebernähe abgeleitet. Jetzt verwendet `_fixture_contexts()` den tatsächlichen Austragungsort des konkreten Spiels:

- Heim-/Auswärtsteam
- tatsächliche Venue-Koordinate
- Ruhetage seit dem vorherigen Spiel
- Reise-/Jetlag-/Höhen-Kontext über `src/context.py`

Dadurch wird z. B. ein Spiel in Mexico City anders bewertet als ein Spiel in Toronto oder Atlanta.

### 2. FIFA-nahe Tie-Breaker

Neu: `src/tiebreakers.py`

Die Gruppensimulation sortiert nicht mehr nur nach Punkten/Tordifferenz, sondern nach einer FIFA-nahen Reihenfolge:

1. Punkte
2. Tordifferenz
3. erzielte Tore
4. Head-to-Head-Minitabelle innerhalb punktgleicher Teams
5. Fallback

Nicht modelliert werden Fair-Play-Punkte und Losentscheid, weil das Projekt keine Karten- oder Losdaten hat. In Monte-Carlo-Läufen wird dafür ein kleiner Zufallsfallback genutzt; in deterministischen Ansichten eine stabile alphabetische Reihenfolge.

### 3. Wahrscheinlichster Turnierbaum

Die deterministische Gruppenreihung für den wahrscheinlichsten Turnierbaum nutzt jetzt zusätzlich erwartete Tore für (`exp_gf`). Dadurch werden Teams mit gleicher erwarteter Punkt-/Tordifferenzlage plausibler sortiert.

### 4. Tests

Neue/angepasste Tests prüfen unter anderem:

- 72 offizielle Gruppenfixtures
- bekannte offizielle Matchdaten und Venues
- Vollständigkeit aller 48 Teams
- Tie-Breaker nach Tordifferenz, erzielten Toren und Head-to-Head
- bestehende Knockout-/Simulationstests

Aktueller Stand:

```text
27 passed
```

## Was weiterhin verbessert werden kann

### 1. Offizielle Kader ersetzen

Sobald die offiziellen WM-Kader veröffentlicht sind, sollte `data/wm2026_squads.csv` ersetzt oder ergänzt werden. Das wäre der größte verbleibende Qualitätshebel.

Empfohlene Mindestfelder:

```text
team,player,position,rating,available
```

Optional sinnvoll:

```text
club,league,age,caps,goals,pace,shooting,passing,dribbling,defending,physical
```

### 2. Karten/Fair-Play-Daten ergänzen

Um FIFA-Tie-Breaker vollständig abzubilden, bräuchte das Projekt Fair-Play-Punkte. Ohne diese bleibt der letzte Tie-Breaker zwangsläufig eine Näherung.

### 3. Kalibrierung mit aktuellen Quoten/Marktwerten prüfen

Für eine produktive Prognose könnten Bookmaker-Implied-Probabilities oder Marktwertdaten als externe Kalibrierung genutzt werden. Das wurde hier nicht eingebaut, weil es Abhängigkeiten, Lizenzfragen und Aktualisierungslogik erhöht.

### 4. CI/Rebuild robuster machen

Die enthaltenen Artefakte sind vorgerechnet und Tests laufen. In sehr kompakten Sandbox-Umgebungen kann ein vollständiger End-to-End-Rebuild nach dem Fit hängen, obwohl die getrennte Postfit-Erzeugung funktioniert. Dafür wurde `--postfit-only` ergänzt, sodass Modell/DB und Vorhersageartefakte getrennt regeneriert werden können.

Empfohlene lokale Nutzung:

```bash
python run.py
python scripts/build_wm2026.py --postfit-only --sims 1000
```

Für Online-Rebuilds:

```bash
python scripts/build_wm2026.py --refresh --sims 1000
```

## Wichtigste Dateien der Überarbeitung

- `src/wm2026.py` – offizieller Spielplan, Gruppen, Teams, Venue-Koordinaten
- `src/tiebreakers.py` – FIFA-nahe Gruppentabellenlogik
- `src/simulation.py` – Simulation nutzt neue Tie-Breaker
- `src/knockout.py` – Turnierbaum nutzt neue Gruppensortierung
- `scripts/build_wm2026.py` – Offline-Fallback und Postfit-Modus
- `data/wm2026_fixtures.csv` – offizieller Gruppenspielplan
- `data/wm2026_predictions.csv` – neu gerechnete Spielprognosen
- `data/wm2026_group_sim.csv` – neu gerechnete Gruppenwahrscheinlichkeiten
- `data/wm2026_knockout_probs.csv` – neu gerechnete Titel-/Rundenchancen
- `data/wm2026_bracket.csv` – neu gerechneter wahrscheinlichster Turnierbaum
