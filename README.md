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

python scripts/init_db.py --fetch  # lädt echte Historie (martj42); ohne --fetch: Beispieldaten
python scripts/train_model.py      # trainiert + gibt Backtest-Metriken aus
streamlit run app.py
```

`--fetch` lädt ~49k echte Länderspiele. Ohne Netzwerk funktioniert alles
auch mit den (synthetischen) Beispieldaten – diese werden bei Bedarf
automatisch erzeugt (`src/sample_data.py`, deterministisch). Wer sie als
CSV braucht: `python scripts/make_sample_data.py`.

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
3. **Dixon-Coles/Poisson-Tor-Modell**: schätzt erwartete Tore beider
   Teams und leitet daraus ein konsistentes Korrektergebnis-Gitter ab –
   Grundlage für 1X2 und die Gruppensimulation.
4. **Logit-Klassifikator** auf Pre-Match-Features (Elo-Differenz, Form,
   Tore, Ruhetage, neutraler Platz).

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

## Projektstruktur

```text
app.py                          Streamlit-Dashboard
scripts/make_sample_data.py     Erzeugt Beispieldaten + WM-2026-Fixtures
scripts/init_db.py              Baut SQLite-DB (--fetch lädt echte Historie)
scripts/train_model.py          Trainiert Modell, speichert es, zeigt Backtest
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
data/sample_matches.csv         Synthetische Beispiel-Historie
data/sample_fixtures_2026.csv   Beispiel-Fixtures (12 Gruppen)
```

> Hinweis: Die mitgelieferte WM-2026-Gruppeneinteilung ist ein
> plausibles Beispiel und sollte vor echtem Einsatz durch die offizielle
> Auslosung ersetzt werden (`data/sample_fixtures_2026.csv`).

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

1. Echte Kaderdaten + erwartete Startelf statt der Näherungs-Demokader.
2. Wetterdaten automatisiert pro Stadion und Anstoßzeit ziehen.
3. Gegen Wettmarktquoten benchmarken (Markt ist ein starker Maßstab).
4. K.-o.-Phase (Verlängerung/Elfmeter) zusätzlich simulieren.
5. Zeitpunktgenaue historische Kader, um die Kaderstärke auch im Backtest
   zu validieren.
