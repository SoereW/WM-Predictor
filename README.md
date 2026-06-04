# WM Predictor

Lauffähiges Fußball-Vorhersage-Dashboard für Länderspiele/WM 2026.
Gibt **kalibrierte Wahrscheinlichkeiten** (1 / X / 2), erwartete Tore (xG)
und Gruppen-Simulationen aus – belegt durch einen Out-of-Sample-Backtest.

## Was enthalten ist

- SQLite-Datenmodell für Spiele und Fixtures, kompatibel zum offenen
  Datensatz `martj42/international_results` (~49.000 echte Spiele ab 1872)
- Hybrides Modell: **World-Football-Elo** + **Kaderstärke (Spieler→Team)**
  + **Dixon-Coles/Poisson-Tor-Modell** + **multinomialer Logit-Klassifikator**,
  kalibriert kombiniert
- Zeitgewichtetes Training (neuere Spiele zählen mehr) und automatische
  Hyperparameter-Wahl auf einem zeitlich abgetrennten Validierungsfenster
- Out-of-Sample-Backtest mit Log-Loss, Brier-Score, Trefferquote und
  Kalibrierungstabelle
- Monte-Carlo-Gruppensimulation (Platzierungswahrscheinlichkeiten)
- Streamlit-Dashboard mit Kontext-Adjustments (Form, Verletzungen,
  Erholung, Wetter)

## Schnellstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# EMPFOHLEN: komplette WM-2026-Prognose mit ECHTEN Daten in einem Befehl
python scripts/build_wm2026.py     # lädt echte Spieler + Historie, baut Tabellen + Vorhersagen
streamlit run app.py               # Dashboard nutzt dieselben echten Daten
```

`build_wm2026.py` lädt echte EA-FC-26-Spielerdaten und die ~49k echten
Länderspiele (martj42), baut je WM-Nation den stärksten verfügbaren Kader,
bewertet jeden Spieler und jedes Team, koppelt das in den Predictor ein und
sagt **alle 72 Gruppenspiele der echten Auslosung** vorher. Details unten
unter „WM 2026 mit echten Daten". Alternativ der klassische Pfad:

```bash
python scripts/init_db.py --fetch  # lädt echte Historie (martj42); ohne --fetch: Beispieldaten
python scripts/train_model.py      # trainiert + gibt Backtest-Metriken aus
streamlit run app.py
```

`--fetch` lädt ~49k echte Länderspiele. Ohne Netzwerk funktioniert alles
auch mit den (synthetischen) Beispieldaten – diese werden bei Bedarf
automatisch erzeugt (`src/sample_data.py`, deterministisch). Wer sie als
CSV braucht: `python scripts/make_sample_data.py`.

## WM 2026 mit echten Daten

`python scripts/build_wm2026.py` ist die End-to-End-Pipeline mit echten,
online geladenen Daten:

1. **Echte Spieler** – EA FC 26 (Stand 2025/26: aktuelle Vereine inkl.
   Sommertransfers 2025, echte Attribute). Je WM-Nation wird der **stärkste
   verfügbare Kader** positionsbewusst zusammengestellt
   (`src/rating/squad_builder.py`).
2. **Ratings** – jeder Spieler bekommt ein Gesamtrating (positionsgewichtetes
   Skill + Form + Erfahrung + Fitness), jedes Team ein Gesamtrating (beste Elf
   + Tiefe + Chemie + Trainer) über das Bewertungssystem `src/rating/`.
3. **Modell** – die ~49k echten Länderspiele (martj42) liefern Elo/Form;
   trainiert wird **strikt vor WM-Beginn (11.06.2026)**, also leak-frei. Die
   Team-Ratings werden als Kaderstärke eingekoppelt.
4. **Vorhersage** – alle 72 Gruppenspiele der **echten Auslosung** (Stand
   Final-Draw 5.12.2025 inkl. Playoff-Sieger) mit 1X2 + xG; je Gruppe eine
   Monte-Carlo-Simulation der Weiterkommens-Wahrscheinlichkeiten.

Ergebnis-Tabellen (im Repo eingecheckt, reproduzierbar):

| Datei | Inhalt |
|-------|--------|
| `data/wm2026_squads.csv` | alle Spieler je Nation mit Roh-Attributen |
| `data/wm2026_player_ratings.csv` | **Spielertabelle mit Gesamtrating** |
| `data/wm2026_team_ratings.csv` | **Team-Rangliste** (Overall, XI, Chemie …) |
| `data/wm2026_fixtures.csv` | echter Gruppenspielplan (72 Spiele) |
| `data/wm2026_predictions.csv` | 1X2 + xG je Spiel, plus Ensemble (`ens_*`) und Reise/Ruhe (`context_elo`) |
| `data/wm2026_group_sim.csv` | P(Platz 1)/P(Top 2) je Team und Gruppe |

> Ehrlichkeit: Die Kader sind der *stärkste verfügbare* Satz echter Spieler
> je Nation laut Datensatz – nicht zwingend die offiziell nominierte
> 26er-Liste. Spieler aus weniger abgedeckten Ligen fehlen in den Quelldaten;
> betroffene Nationen (z. B. Iran, Jordanien, Usbekistan) haben dünnere Kader,
> ausgewiesen über `n_players`. Die Auslosung wurde per Web-Recherche
> zusammengetragen (`src/wm2026.py`) – vor produktivem Einsatz gegen die
> offizielle FIFA-Quelle abgleichen.

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
   Kaderstärke das effektive Rating moderat (Standard 35 %).
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
Elo-Korrektur (moderat, `squad_pull`, Standard 35 %). Wie die einfache
Kaderstärke greift er **erst zur Vorhersage** – die trainierten ML-Modelle
sehen ihn nie und können nicht overfitten.

**Score- vs. Historie-System im direkten Vergleich:** `python
scripts/compare_systems.py` misst beide Stärke-Signale out-of-sample an echten
Spielen (Standard: 2024 bis WM-Start, nur WM-Nationen). Typisches Bild:

| System | Log-Loss ↓ | Brier ↓ | Treffer ↑ |
|--------|-----------:|--------:|----------:|
| Basisrate (naiv) | 1.106 | 0.671 | 40.8 % |
| Score (Kader, EA FC 26) | 1.043 | 0.628 | 46.2 % |
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

**Echte Spielerdaten:** `src/rating/fifa_ingest.py` lädt einen offenen
EA-SPORTS-FC-Spielerdatensatz (echte Attribute) und übersetzt ihn ins
Bewertungs-Schema. Standardquelle ist **EA FC 26** (Stand 2025/26 – echte,
aktuelle Werte); der ältere FIFA-20-Datensatz bleibt als Fallback. Eigene
Kaderdaten via `src/rating/io.load_squads_csv` (Format: `team, player,
position` + Attribute oder `overall`; optional `club, league, age, caps,
form, available`).

## Projektstruktur

```text
app.py                          Streamlit-Dashboard (nutzt echte Daten, falls vorhanden)
scripts/build_wm2026.py         End-to-End-Pipeline WM 2026 mit echten Daten
scripts/make_sample_data.py     Erzeugt Beispieldaten + WM-2026-Fixtures
scripts/init_db.py              Baut SQLite-DB (--fetch lädt echte Historie)
scripts/train_model.py          Trainiert Modell, speichert es, zeigt Backtest
scripts/evaluate_ratings.py     Stichproben-Auswertung des Bewertungssystems
scripts/validate_predictions.py Vorhersage vs. echtes Ergebnis (out-of-sample)
scripts/compare_systems.py      Score- vs. Historie-System vergleichen (out-of-sample)
src/wm2026.py                   Echte Gruppen-Auslosung + Spielplan-Generator
src/database.py                 DB-Schema und CSV-Ingestion
src/teams.py                    Normalisierung von Teamnamen
src/elo.py                      World-Football-Elo
src/features.py                 Feature-Engineering (leak-frei)
src/model.py                    Hybrid-Modell: Training + Prediction
src/squad.py                    Kaderstärke: Spieler→Team, Elo-Kalibrierung
src/sample_squads.py            Deterministische Demo-Kaderdaten (Näherung)
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
src/rating/fifa_ingest.py       Echte EA-FC-Spielerdaten → Bewertungs-Schema
tests/test_rating.py            Unit-Tests des Bewertungssystems
tests/test_context_tactics.py   Unit-Tests für Kontext- und Taktik-Faktoren
tests/test_wm2026.py            Unit-Tests für Auslosung, Spielplan, Kaderbau
data/wm2026_player_ratings.csv  Alle Spieler je Nation mit Gesamtrating
data/wm2026_team_ratings.csv    Team-Rangliste (Overall, XI, Tiefe, Chemie)
data/wm2026_predictions.csv     1X2 + xG für alle 72 Gruppenspiele
data/wm2026_group_sim.csv       Weiterkommens-Wahrscheinlichkeiten je Gruppe
```

> Hinweis: `src/wm2026.py` enthält die echte WM-2026-Auslosung (Final-Draw
> 5.12.2025 inkl. Playoff-Sieger), per Web-Recherche zusammengetragen. Vor
> produktivem Einsatz gegen die offizielle FIFA-Quelle abgleichen – eine
> Änderung in `GROUPS` genügt, der Spielplan wird daraus generiert.

## Datenquellen zum Anschließen

- Historische Länderspiele: `martj42/international_results` (GitHub, CC0) –
  bereits per `--fetch` integriert.
- WM-Spielplan: FIFA-Webseite oder gepflegte CSV (Format wie
  `sample_fixtures_2026.csv`).
- Wetter: Open-Meteo (`src/weather.py`, Erweiterungspunkt).
- Kaderdaten: eigene CSV (`team, player, position, rating[, available]`) –
  echte Spielerratings (z. B. SoFIFA/EA-Scrapes) oder Marktwerte als Proxy.
- Quoten: gut als Benchmark, Lizenzbedingungen beachten.

## Nächste Schritte zur weiteren Verbesserung

1. ✅ Echte Spielerdaten (EA FC 26) + echte Auslosung integriert
   (`scripts/build_wm2026.py`). Offen: die offiziell **nominierte** 26er-Liste
   bzw. erwartete Startelf je Spiel statt des stärksten verfügbaren Kaders.
2. Wetterdaten automatisiert pro Stadion und Anstoßzeit ziehen.
3. Gegen Wettmarktquoten benchmarken (Markt ist ein starker Maßstab).
4. K.-o.-Phase (Verlängerung/Elfmeter) zusätzlich simulieren.
5. Zeitpunktgenaue historische Kader, um die Kaderstärke auch im Backtest
   zu validieren.
