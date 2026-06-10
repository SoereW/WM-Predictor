# WM 2026 Predictor

Lauffähiges Vorhersage-Dashboard für die **WM 2026**: kalibrierte
Wahrscheinlichkeiten (1 / X / 2), erwartete Tore (xG), **Titelchancen jeder
Nation**, ein **kompletter Turnierbaum (K.-o.-Phase)** und alle 72 Gruppen-
sowie alle K.-o.-Spiele – plus eine optionale Eingabe für **aktuelle
Ereignisse (Verletzungen/Sperren)**.

> Der **aktuelle Kader** zählt bewusst stärker als alte Länderspielergebnisse:
> ein 3:0 von vor zehn Jahren sagt wenig über die Elf von morgen. Die Stärke
> kommt aus den mitgelieferten Kader-/Spielerdaten und wird durch robuste
> Historie/Elo korrigiert. Wichtig: Die Kader sind Näherungen, keine offiziell
> nominierten WM-Kader.

## Schnellstart (ein Befehl)

```bash
python run.py
```

Das war's. `run.py` installiert bei Bedarf die Abhängigkeiten, baut beim
**ersten** Start aus den **eingecheckten Daten** (kein Download nötig) die
Datenbank und das Modell und öffnet das Dashboard im Browser
(http://localhost:8501). Alternativ klassisch:

```bash
pip install -r requirements.txt
streamlit run app.py               # App baut fehlende Artefakte selbst
```

**Alle Daten sind im Repo** (`data/`): die bereinigte Historie
(`international_results.csv`, ~49k Länderspiele), die fertig bewerteten Kader
inkl. aller Attribute (`wm2026_squads.csv`), Team-/Spieler-Ratings, der echte
offizielle Gruppenspielplan, die Gruppenvorhersagen, Titelchancen und der Turnierbaum. Es läuft
also offline sofort. Fehlt die Historie einmal, lädt der Bootstrap sie beim
ersten Start automatisch (offline-Fallback: deterministische Beispieldaten).

## Was diese Überarbeitung gefixt & verbessert hat

- **Startfehler behoben:** `app.py`/`run.py` verwiesen auf `src/bootstrap.py`
  und `src/knockout.py`, die im Upload fehlten. Beide Module sind nun ergänzt
  (Erststart-Bootstrap bzw. komplette K.-o.-Phase), ebenso die vom Dashboard
  genutzten Funktionen `WMPredictor.matchup_from_states` und
  `wm2026.fixture_venue_coord`.
- **Echte Daten:** Beim ersten `python run.py` wird die echte Länderspiel-
  Historie geladen, eine leak-freie Modellbasis trainiert und alle Artefakte
  (Vorhersagen, Gruppen, Titelchancen, Turnierbaum) erzeugt – **mit einem
  Befehl**, falls noch nicht vorhanden, dann Start des Dashboards.
- **Turnierbaum bis zum Finale:** Der Baum ist vollständig durchgerechnet
  (Sechzehntelfinale → Finale). **Auf ein Spiel zeigen** zeigt die Kurzinfo,
  **klicken** (oder im Spiel-Center auswählen) öffnet die Detail-Prognose
  (1X2, xG, wahrscheinlichste Ergebnisse, Weiterkommens-Chance, Modell-Faktoren).
- **Prognostizierter Endstand je Spiel:** Jedes Spiel – Gruppe **und** K.-o. –
  zeigt das wahrscheinlichste Ergebnis (z. B. `2:1`): im Turnierbaum, in den
  Tabellen „Alle Spiele" und in der Detail-Prognose. Bei K.-o.-Spielen wird ein
  **entschiedener** Endstand passend zum weiterkommenden Team gezeigt.
- **Kaderstärke & Chemie zählen stark:** Standardgewicht jetzt **70 %**
  (Regler bis 90 %). Der Team-Score (beste Elf + Tiefe + **Chemie** + Trainer)
  dominiert die Prognose; je Spiel ist der direkte Kader-/Chemie-Vergleich beider
  Teams sichtbar.
- **Schlicht, modern, clean:** helles Theme, ruhige Typografie, ein klar
  lesbarer Turnierbaum als Mittelpunkt.

### Alles neu mit frischen Online-Daten bauen

```bash
python scripts/build_wm2026.py --postfit-only --sims 1000  # nutzt vorhandenes Modell/DB und baut Prognose-Artefakte neu
python scripts/build_wm2026.py --squad-pull 0.6              # kompletter Rebuild, Kader stärker gewichten
python scripts/build_wm2026.py --refresh                    # Online-Quellen bewusst neu laden
```

`build_wm2026.py` nutzt vorhandene Kaderdaten offline. Liegt
`data/fc26_players.csv` vor oder wird `--refresh` verwendet, kann die Pipeline
Roh-Spielerdaten neu einlesen; sonst wird reproduzierbar mit
`data/wm2026_squads.csv` gearbeitet. Die Historie wird leak-frei vor
WM-Beginn trainiert; anschließend entstehen **alle Gruppenvorhersagen, die
komplette Turniersimulation und der wahrscheinlichste Turnierbaum**.



## Überarbeitung vom 10.06.2026

Dieses Paket wurde geprüft und gegenüber dem ursprünglichen Upload überarbeitet:

- `data/wm2026_fixtures.csv` und `src/wm2026.py` enthalten jetzt den offiziellen 72-Spiele-Gruppenspielplan inklusive Austragungsort je Spiel. Die vorherige Version nutzte einen vereinfachten, nicht offiziellen Generator pro Gruppe.
- `data/international_results.csv` wurde bereinigt: zukünftige WM-Fixtures ohne Ergebnis wurden aus der Historie entfernt. Trainingsbasis: 49.339 abgeschlossene Länderspiele, letzter Eintrag 2026-06-05.
- Der Spielkontext nutzt jetzt die tatsächlichen Venue-Koordinaten je Fixture statt nur einer Gruppen-/Gastgeber-Näherung.
- Gruppentabellen und Turnier-Simulation verwenden FIFA-nahe Tie-Breaker: Punkte, Tordifferenz, erzielte Tore, Head-to-Head-Minitabelle; Fair-Play und Losentscheid werden mangels Karten-/Losdaten durch deterministische bzw. Monte-Carlo-Fallbacks ersetzt.
- Neue Tests decken offizielle Fixtures und Tie-Breaker ab; Stand dieser Version: `27 passed`.

Details stehen in `DATA_AUDIT.md`.

## WM 2026 mit echten Daten

`python scripts/build_wm2026.py` ist die End-to-End-Pipeline. Sie läuft
offline mit den mitgelieferten Daten und kann bei Bedarf Online-Quellen neu
einlesen:

1. **Kader/Spieler** – offline über `data/wm2026_squads.csv`; optional aus
   `data/fc26_players.csv`/Online-Refresh neu baubar. Je WM-Nation wird der
   **stärkste verfügbare Kader** positionsbewusst zusammengestellt
   (`src/rating/squad_builder.py`).
2. **Ratings** – jeder Spieler bekommt ein Gesamtrating (positionsgewichtetes
   Skill + Form + Erfahrung + Fitness), jedes Team ein Gesamtrating (beste Elf
   + Tiefe + Chemie + Trainer) über das Bewertungssystem `src/rating/`.
3. **Modell** – die 49.339 abgeschlossenen Länderspiele liefern Elo/Form;
   trainiert wird **strikt vor WM-Beginn (11.06.2026)**, also leak-frei. Die
   Team-Ratings werden als Kaderstärke eingekoppelt.
4. **Vorhersage** – alle 72 offiziellen Gruppenspiele laut FIFA-Spielplan
   (Stand Prüfung 10.06.2026) mit 1X2 + xG; je Gruppe eine
   Monte-Carlo-Simulation der Weiterkommens-Wahrscheinlichkeiten.
5. **Turnier** – eine komplette Monte-Carlo-Simulation des gesamten Turniers
   (Gruppen → Finale) liefert je Nation die Chance, jede Runde zu erreichen und
   **Weltmeister** zu werden; zusätzlich der **wahrscheinlichste Turnierbaum**.

Ergebnis-Tabellen (im Repo eingecheckt, reproduzierbar):

| Datei | Inhalt |
|-------|--------|
| `data/wm2026_squads.csv` | alle Spieler je Nation mit Roh-Attributen |
| `data/wm2026_player_ratings.csv` | **Spielertabelle mit Gesamtrating** |
| `data/wm2026_team_ratings.csv` | **Team-Rangliste** (Overall, XI, Chemie …) |
| `data/wm2026_fixtures.csv` | offizieller Gruppenspielplan (72 Spiele) |
| `data/wm2026_predictions.csv` | 1X2 + xG je Spiel, plus Reise/Ruhe (`context_elo`) |
| `data/wm2026_group_sim.csv` | P(Platz 1)/P(Top 2) je Team und Gruppe |
| `data/wm2026_knockout_probs.csv` | **Titelchancen**: P(Achtel…Finale, Titel) je Nation |
| `data/wm2026_bracket.csv` | **wahrscheinlichster Turnierbaum** (alle K.-o.-Spiele) |

> Ehrlichkeit: Die Kader sind der *stärkste verfügbare* Satz echter Spieler
> je Nation laut Datensatz – nicht zwingend die offiziell nominierte
> 26er-Liste. Spieler aus weniger abgedeckten Ligen fehlen in den Quelldaten;
> betroffene Nationen (z. B. Iran, Jordanien, Usbekistan) haben dünnere Kader,
> ausgewiesen über `n_players`. Der Gruppenspielplan wurde am 10.06.2026 gegen die offizielle
> FIFA-Fixture-Liste abgeglichen und in `src/wm2026.py` fest hinterlegt.

## Turnierbaum & Titelchancen (K.-o.-Phase)

Das neue WM-Format: 12 Gruppen à 4 Teams; weiter kommen die **zwei Besten je
Gruppe** plus die **acht besten Gruppendritten** → 32 Teams in einer reinen
K.-o.-Runde (Sechzehntel- → Achtel- → Viertel- → Halbfinale → Finale).

`src/knockout.py` enthält das **offizielle FIFA-Bracket** (Spiele 73–104 des
veröffentlichten Spielplans): welcher Gruppensieger/-zweite in welchem Spiel
antritt, ist vorab festgelegt; nur die Zuordnung der acht Dritten hängt davon
ab, aus welchen Gruppen sie kommen (FIFA-Kombinationstabelle, Annex C). Diese
Zuordnung wird über ein **regelkonformes Matching** der erlaubten Gruppen-Sets
nachgebildet (ein Dritter trifft nie auf seinen eigenen Gruppensieger). Ein
Test prüft, dass für **alle 495** möglichen Drittel-Konstellationen ein
gültiges Bracket entsteht.

- **Turnier-Simulation** (Monte Carlo): simuliert Gruppen **und** K.-o. in einem
  Durchlauf und liefert je Nation P(Achtelfinale … Finale, **Titel**). K.-o.-
  Spiele kennen kein Remis – der Remis-Anteil wird über die relative Stärke
  (Verlängerung/Elfmeter) aufgeteilt.
- **Wahrscheinlichster Turnierbaum**: deterministischer Baum aus den Erwartungs-
  werten (erwartete Gruppenpunkte → Favorit kommt je K.-o.-Spiel weiter), im
  Dashboard als grafischer Baum.

Beides ist effizient: Team-Zustände und alle paarweisen Wahrscheinlichkeiten
werden **einmal** vorberechnet, danach ist jeder Durchlauf reines Sampling.

## Aktuelle Ereignisse eintragen (Verletzungen/Sperren)

Im Dashboard (Seitenleiste → „Aktuelle Ereignisse") lassen sich Spieler in
einem einfachen Format eintragen – **eine Zeile pro Spieler**:

```text
Spain; Rodri                 # fehlt → nicht verfügbar (aus dem Kader genommen)
France; Mbappe; angeschlagen # spielt, aber Form −15 %
England; Bellingham; out
```

`Team; Spielername` (optionaler Status `out` / `angeschlagen`). Die Namens-
suche ist tolerant (Groß/Klein, Akzente). Daraufhin werden die betroffenen
Team-Ratings **neu berechnet** und alle Vorhersagen, Titelchancen und der
Turnierbaum aktualisiert. Über den Regler **„Kaderstärke-Gewicht"** lässt sich
zudem einstellen, wie stark der aktuelle Kader gegenüber der Historie zählt.

## Vorhersagequalität (Out-of-Sample-Backtest)

Training auf allen Spielen bis ~2018, Test auf **7.389 echten Spielen
(2018–2026)**, die das Modell nie gesehen hat:

| Modell                | Log-Loss ↓ | Brier ↓ | Trefferquote ↑ |
|-----------------------|-----------:|--------:|---------------:|
| Basisrate (naiv)      |     1.0505 |  0.6335 |         47,8 % |
| Nur Elo               |     0.8652 |  0.5085 |         60,5 % |
| **Hybrid (dieses Repo)** | **0.8602** | **0.5051** |     **60,7 %** |

**Kalibrierung** (vorhergesagte vs. tatsächlich eingetretene Favoritenquote):

| Vorhergesagt | Tatsächlich | n     |
|-------------:|------------:|------:|
| 0,45         | 0,45        | 1 749 |
| 0,65         | 0,68        | 1 264 |
| 0,75         | 0,77        |   976 |
| 0,85         | 0,87        |   766 |
| 0,94         | 0,95        |   358 |

Sagt das Modell „80 %", tritt das Ergebnis real in ~87 % der Fälle ein –
die ausgegebenen Wahrscheinlichkeiten sind also verlässlich, nicht nur die
Tipp-Richtung.

## Modellidee

Bewusst hybrid statt eines reinen Deep-Learning-Modells – WM-Daten sind
klein und verrauscht, und im Backtest schlägt der ML-Aufbau reines Elo nur
knapp. Der Hebel liegt daher nicht in mehr Modellkomplexität, sondern in
besseren **Features** – insbesondere der Kaderstärke.

1. **Elo-Rating** als robuste Stärke-Baseline (turnierabhängiger
   K-Faktor, Tordifferenz-Multiplikator, Heimvorteil, sequenziell und
   damit leak-frei).
2. **Kaderstärke (Spieler → Team)**: Jeder Spieler wird bewertet, positions­
   bewusst zur Teamstärke aggregiert (beste Elf + Kadertiefe) und auf die
   Elo-Skala kalibriert. Elo misst nur, *wie eine Nation historisch
   gespielt hat* – nicht, *wie stark der Kader ist, der morgen aufläuft*.
   Genau dort (Kaderumbruch, junge Generation, lange Pausen) korrigiert die
   Kaderstärke das effektive Rating. **Standard 50 %** (`squad_pull`): alte
   Länderspielergebnisse hängen nur an den Spielern von damals, deshalb zählt
   für die WM-Prognose der **aktuelle Kader bewusst stark**. Im Dashboard frei
   einstellbar; dünn abgedeckte Nationen werden über die Datenabdeckung
   (`coverage`) vorsichtiger eingekoppelt, damit fehlende Spieler sie nicht
   unfair abwerten.
3. **Gegnerbezogene Form** (trainiert & backtestbar): nicht nur *ob* ein
   Team zuletzt gewann, sondern *gegen wen* (`sos` – mittlere Gegnerstärke
   des Spielplans) und *ob über/unter Erwartung* (`form_vs_exp` – erzielte
   minus aus der Elo-Differenz erwartete Punkte). Beides leak-frei aus der
   Historie, senkt den Out-of-Sample-Log-Loss messbar (0.8602 → 0.8596).
4. **Spielkontext** (`src/context.py`, Vorhersagezeitpunkt): Ruhetage/Pause,
   Reisedistanz + Jetlag, Stadionhöhe (Akklimatisierung) und Hitze/Klima –
   jeweils als gewichtete, **gedeckelte** Elo-Korrektur.
5. **Taktik-Matchup** (`src/rating/tactics.py`): Taktikprofil je Team
   (Pressing, Ballbesitz, Tempo, Block, Flügel, Konter) und gewichtete
   Stil-Regeln (z. B. Konter schlägt Hochpressing) → kleine Elo-Korrektur.
6. **Dixon-Coles/Poisson-Tor-Modell**: schätzt erwartete Tore beider
   Teams und leitet daraus ein konsistentes Korrektergebnis-Gitter ab –
   Grundlage für 1X2 und die Gruppensimulation.
7. **Logit-Klassifikator** auf Pre-Match-Features (Elo-Differenz, Form,
   Tore, Ruhetage, gegnerbezogene Form, neutraler Platz).

Faktoren 2/4/5 greifen wie die Kaderstärke **erst zur Vorhersage** als
Elo-Korrektur – die trainierten ML-Modelle sehen sie nie und können nicht
overfitten. Faktoren 1/3 sind echte Trainingsfeatures und im Backtest belegt.

Tor-Modell und Klassifikator werden gewichtet gemischt; Mischgewicht und
die Dixon-Coles-Korrektur `rho` werden auf einem zeitlich abgetrennten
Validierungsfenster auf **minimalen Log-Loss** optimiert.

**Warum die Kaderstärke nicht overfittet:** Sie greift ausschließlich zum
Vorhersagezeitpunkt als Elo-Korrektur – die trainierten ML-Modelle sehen
sie nie. Sie kann also nichts „auswendig lernen", sondern nur den robusten
Elo-Input verschieben. Eigene Kaderdaten lassen sich per CSV einspeisen
(Format: `team, player, position, rating[, available]`); verletzte/gesperrte
Spieler werden über `available=0` ausgeschlossen.

> Hinweis: Die mitgelieferten Kaderdaten sind **Näherungen** und werden
> deterministisch erzeugt (`src/sample_squads.py`). Sie bilden den
> *aktuellen* Stand ab und fließen daher bewusst **nicht** in den
> historischen Backtest ein (das wäre anachronistisch) – sie verbessern die
> WM-2026-Prognose, nicht die Backtest-Zahl. Ein sauberer Kader-Backtest
> bräuchte zeitpunktgenaue historische Kader.

Im Dashboard lassen sich zusätzlich Kontext-Adjustments (Startelf,
Verletzungen, Erholung, Wetter) als Elo-Zuschlag einstellen.

## Bewertungssystem (Spieler / Team / Chemie)

Eigenständiges Subsystem (`src/rating/`), **getrennt** vom Predictor. Es
beantwortet die Frage „wie stark ist dieser Kader und wie gut passt er
zusammen?" anhand klar definierter, **gewichteter Kriterien** – die zentrale
Stellschraube ist `src/rating/criteria.py`.

**Spieler-Gesamtscore** (0–100):
- *Skill* (Gewicht 0,68): Rohattribute (Tempo, Abschluss, Passspiel,
  Technik, Defensive, Physis, Übersicht, Torwart) werden mit dem
  **Positionsprofil** der Rolle gewichtet – ein Stürmer an Abschluss/Tempo,
  ein Innenverteidiger an Zweikampf/Physis.
- *Form* (0,16), *Erfahrung* (0,10, aus Länderspielen + Alter),
  *Fitness/Verfügbarkeit* (0,06).

**Team-Gesamtscore** (0–100): beste Elf (0,62) + Kadertiefe (0,13) +
**Chemie** (0,17) + Trainer (0,08).

**Chemie** als eigener Faktor, fünf gewichtete Teilkriterien:
Vereinsblöcke (liga-gewichtet), Eingespieltheit (gemeinsame Caps),
Positionstreue, Altersbalance, Trainer-Kontinuität. Wichtig: Chemie wirkt
**ans Niveau gekoppelt** – perfekte Chemie macht aus einem schwachen Kader
keinen Favoriten, hebt aber einen starken Kader spürbar.

**Validierung per Stichprobe** (statt Backtest – es gibt keine „wahre"
Chemie-Zahl): `python scripts/evaluate_ratings.py` bewertet bekannte Spieler
und Teams und prüft automatisch Plausibilität (Top-Spieler oben, Top-Nation
vorn, Chemie nicht überzeichnet). Unit-Tests: `python tests/test_rating.py`.

**Validierung an echten Spielen** (Mannschaften/Spieler vergleichen →
Vorhersage → mit Endergebnis abgleichen): `python scripts/validate_predictions.py`.
Für jedes Testspiel wird ein frisches Modell **nur auf Spielen davor**
trainiert (out-of-sample, kein Leakage – echte Prognose). Optionen:
`--real-players` nutzt echte FIFA-Spielerattribute statt der Demo-Kader,
`--insample` zum Vergleich das fertige Gesamtmodell.

**Einkopplung in den Predictor:** Der Team-Gesamtscore (Spieler + Chemie +
Trainer) speist den Predictor über `WMPredictor.attach_team_scores` als
Elo-Korrektur (`squad_pull`, Standard 50 % – aktueller Kader zählt stark).
Wie die einfache Kaderstärke greift er **erst zur Vorhersage** – die
trainierten ML-Modelle sehen ihn nie und können nicht overfitten.

**Score- vs. Historie-System im direkten Vergleich:** `python
scripts/compare_systems.py` misst beide Stärke-Signale out-of-sample an echten
Spielen (Standard: 2024 bis WM-Start, nur WM-Nationen). Typisches Bild:

| System | Log-Loss ↓ | Brier ↓ | Treffer ↑ |
|--------|-----------:|--------:|----------:|
| Basisrate (naiv) | 1.106 | 0.671 | 40.8 % |
| Score (Kader-/Spielerdaten) | 1.043 | 0.628 | 46.2 % |
| Historie (ML, ~49k Spiele) | 1.067 | 0.649 | 44.5 % |
| Hybrid (Score→Elo→ML) | 1.049 | 0.636 | **45.9 %** |
| Hybrid + Coverage-Konfidenz | 1.052 | 0.638 | 45.2 % |
| Hybrid + Ruhetage | 1.049 | 0.636 | 45.9 % |
| Ensemble Ø(Score, Historie) | **1.039** | **0.628** | 44.2 % |

Beide Systeme schlagen die Basisrate und sind **komplementär** (Tipps stimmen
nur zu ~75 % überein): der Kader bildet *Talent* ab, die Historie *Resultate/
Form*. Das **Ensemble** liefert die beste Wahrscheinlichkeits-Qualität
(Log-Loss/Brier), der Hybrid die beste Trefferquote. Das Skript zeigt zudem,
**wo** sich die Systeme uneinig sind (z. B. unterschätzt der Kader Nationen mit
vielen Heimliga-Spielern wie Japan/Iran, deren Spieler im Datensatz fehlen).

**Was die Score-Analyse zusätzlich nutzt** (jeweils am Vergleichs-Harness
geprüft – es bleibt nur, was misst):

- **Positionsgewichtung** (`ROLE_IMPORTANCE`, `STAR_ALPHA` in
  `criteria.py`): die Achse (TW/IV/Sechser/Stürmer) und die stärksten Spieler
  zählen etwas mehr – „wie es in echt ist". Senkt den Score-Log-Loss leicht und
  konsistent (1.0433 → 1.0426) und ist jetzt Standard.
- **Reiseweg + Ruhetage** (`src/context.py`, auto pro Spiel): interkontinental
  weit gereiste Teams sind leicht im Nachteil, Gastgeber lokal im Vorteil;
  Ruhetage aus dem Spielplan-Abstand. Greift „immer" in den WM-Vorhersagen
  (Spalte `context_elo`). Aggregat-Effekt klein (Belastung ist meist
  symmetrisch), aber football-korrekt für asymmetrische Fälle.
- **Ensemble** (Spalten `ens_*` in `wm2026_predictions.csv`): Mittel aus
  Score- und Hybrid-Vorhersage – die im Backtest beste Wahrscheinlichkeits-
  Qualität.
- **Coverage-Konfidenz** (optional, `--coverage`): bei dünn abgedeckten Kadern
  (wenige Spieler im Datensatz) wird die Kaderstärke vorsichtiger eingekoppelt.
  Ehrlich: senkt den Log-Loss auf aktuellen Daten *nicht* (es war eher besser,
  dem Kader zu vertrauen) – daher standardmäßig aus, aber als robuste Option
  vorhanden.

**Spieler-/Kaderdaten:** `src/rating/fifa_ingest.py` kann einen offenen
EA-SPORTS-FC-/FIFA-Spielerdatensatz einlesen und ins Bewertungsschema
übersetzen. In diesem Paket ist der große Rohdatensatz `data/fc26_players.csv`
nicht enthalten; offline nutzt die Pipeline deshalb `data/wm2026_squads.csv`.
Eigene Kaderdaten via `src/rating/io.load_squads_csv` (Format: `team, player,
position` + Attribute oder `overall`; optional `club, league, age, caps, form,
available`).

## Projektstruktur

```text
run.py                          Ein-Befehl-Starter (Deps + Build + Dashboard)
app.py                          Streamlit-Dashboard (Titelchancen, Turnierbaum, alle Spiele)
src/bootstrap.py                Erststart: DB + Modell aus eingecheckten Echtdaten (offline)
src/knockout.py                 K.-o.-Bracket + Turnier-Simulation + bester Baum
scripts/build_wm2026.py         Pipeline WM 2026; offline/online rebuildbar
scripts/make_sample_data.py     Erzeugt Beispieldaten
scripts/init_db.py              Baut SQLite-DB (--fetch lädt echte Historie)
scripts/train_model.py          Trainiert Modell, speichert es, zeigt Backtest
scripts/evaluate_ratings.py     Stichproben-Auswertung des Bewertungssystems
scripts/validate_predictions.py Vorhersage vs. echtes Ergebnis (out-of-sample)
scripts/compare_systems.py      Score- vs. Historie-System vergleichen (out-of-sample)
src/wm2026.py                   Gruppen + offizieller 72-Spiele-Gruppenspielplan
src/database.py                 DB-Schema und CSV-Ingestion
src/teams.py                    Normalisierung von Teamnamen
src/elo.py                      World-Football-Elo
src/features.py                 Feature-Engineering (leak-frei)
src/model.py                    Hybrid-Modell: Training + Prediction
src/squad.py                    Kaderstärke: Spieler→Team, Elo-Kalibrierung
src/sample_squads.py            Deterministische Demo-/Fallback-Kaderdaten
src/simulation.py               Monte-Carlo-Gruppensimulation
src/backtest.py                 Out-of-Sample-Backtest + Kalibrierung
src/ingest.py                   Download offener Datenquellen (martj42)
src/weather.py                  Optionale Open-Meteo-Integration
src/context.py                  Kontext: Reise/Pause/Höhe/Klima (Elo-Korrektur)
src/rating/tactics.py           Taktikprofile + Stil-Matchups
src/rating/criteria.py          Kriterien, Positionsprofile, Gewichte (zentral)
src/rating/player.py            Spielerbewertung (Attribute→Skill→Score)
src/rating/chemistry.py         Team-Chemie (5 gewichtete Teilkriterien)
src/rating/team.py              Team-Aggregation inkl. Chemie + Trainer
src/rating/sample.py            Kuratierte bekannte Spieler (für Stichproben)
src/rating/squad_builder.py     Stärksten Kader je Nation aus Spielerpool bauen
src/rating/io.py                Loader für echte Kader-CSVs
src/rating/fifa_ingest.py       EA-FC-/FIFA-Rohdaten → Bewertungs-Schema
tests/test_rating.py            Unit-Tests des Bewertungssystems
tests/test_context_tactics.py   Unit-Tests für Kontext- und Taktik-Faktoren
tests/test_wm2026.py            Unit-Tests für Gruppen, offizielle Fixtures, Kaderbau
tests/test_knockout.py          Unit-Tests für Bracket + Drittel-Zuordnung (alle 495 Fälle)
data/international_results.csv   Bereinigte Historie (49.339 abgeschlossene Länderspiele)
data/wm2026_player_ratings.csv  Alle Spieler je Nation mit Gesamtrating
data/wm2026_team_ratings.csv    Team-Rangliste (Overall, XI, Tiefe, Chemie)
data/wm2026_predictions.csv     1X2 + xG für alle 72 Gruppenspiele
data/wm2026_group_sim.csv       Weiterkommens-Wahrscheinlichkeiten je Gruppe
data/wm2026_knockout_probs.csv  Titelchancen je Nation (P Achtelfinale…Titel)
data/wm2026_bracket.csv         Wahrscheinlichster Turnierbaum (alle K.-o.-Spiele)
```

> Hinweis: `src/wm2026.py` enthält Gruppen und offiziellen Gruppenspielplan
> mit Stand der Prüfung vom 10.06.2026. Bei einer FIFA-Änderung genügt eine
> Anpassung von `GROUPS`/`OFFICIAL_GROUP_FIXTURES`; die CSV-Artefakte können
> danach neu generiert werden.

## Datenquellen zum Anschließen

- Historische Länderspiele: `martj42/international_results` (GitHub, CC0) –
  bereits per `--fetch` integriert.
- WM-Spielplan: in `src/wm2026.py` fest hinterlegt; bei Änderungen gegen FIFA-Fixtures prüfen.
- Wetter: Open-Meteo (`src/weather.py`, Erweiterungspunkt).
- Kaderdaten: eigene CSV (`team, player, position, rating[, available]`) –
  echte Spielerratings (z. B. SoFIFA/EA-Scrapes) oder Marktwerte als Proxy.
- Quoten: gut als Benchmark, Lizenzbedingungen beachten.

## Nächste Schritte zur weiteren Verbesserung

1. ✅ Offizieller Gruppenspielplan integriert. Offen: offizielle **nominierte**
   26er-Kader bzw. erwartete Startelf je Spiel statt des stärksten verfügbaren
   Kaders.
2. ✅ Komplette K.-o.-Phase (Bracket, Verlängerung/Elfmeter) +
   Turnierbaum + Titelchancen (`src/knockout.py`).
3. ✅ Eingabe aktueller Ereignisse (Verletzungen/Sperren) im Dashboard.
4. Wetterdaten automatisiert pro Stadion und Anstoßzeit ziehen.
5. Gegen Wettmarktquoten benchmarken (Markt ist ein starker Maßstab).
6. Zeitpunktgenaue historische Kader, um die Kaderstärke auch im Backtest
   zu validieren.
