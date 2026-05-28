# WM Predictor Prototype

Ein lauffähiger Prototyp für ein Fußball-WM-Vorhersage-Dashboard.

## Was enthalten ist

- SQLite-Datenmodell für Spiele, Fixtures, Teamratings, Spielerform und Wetter
- Daten-Ingestion für Beispiel-Daten und optionale offene Datenquellen
- Basismodell: Elo + Machine-Learning-Klassifikator
- Streamlit-Dashboard für Spielvorhersagen und Sensitivitätsanalyse
- Erweiterungspunkte für Spielerform, Verletzungen, Quoten, Wetter und erwartete Startelf

## Schnellstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/init_db.py
python scripts/train_model.py
streamlit run app.py
```

Das Dashboard nutzt zuerst die Beispiel-Daten in `data/`. Für echte Prognosen solltest du die offenen historischen Ergebnisse laden und anschließend Spieler-/Lineup-Daten ergänzen.

## Datenquellen, die du anschließen kannst

- Historische Länderspiele: `martj42/international_results` auf GitHub oder Kaggle.
- WM-Spielplan: FIFA-Webseite oder manuell gepflegte CSV.
- Wetter: Open-Meteo oder Meteostat.
- Spielerdaten: kostenpflichtig meist besser, z. B. StatsBomb/Opta/Wyscout; freie Alternativen sind lückenhaft.
- Quoten: je nach Anbieter/API; gut als Benchmark, aber Lizenzbedingungen beachten.

## Modellidee

Der Prototyp ist absichtlich hybrid:

1. Elo-Rating als robuste Baseline.
2. ML-Modell auf Features wie Elo-Differenz, Form, Tore, neutraler Platz, Spielkontext.
3. Manuelle Kontext-Adjustments im Dashboard für Spielerform, Wetter, Erholung und Ausfälle.

Das ist praktischer als direkt ein komplexes Deep-Learning-Modell, weil WM-Daten klein und verrauscht sind. Erst wenn sehr viele hochwertige Spieler-, Event- und Clubdaten integriert sind, lohnt sich ein komplexeres KI-Modell.

## Projektstruktur

```text
app.py                         Streamlit-Dashboard
scripts/init_db.py             Erstellt SQLite-DB aus CSVs
scripts/train_model.py         Trainiert Modell und speichert Artefakte
src/database.py                DB-Schema und Laden von CSVs
src/elo.py                     Elo-Berechnung
src/features.py                Feature Engineering
src/model.py                   Training und Prediction
src/weather.py                 Open-Meteo-Integration
src/simulation.py              Turnier-/Gruppen-Simulation, Startpunkt
data/sample_matches.csv        Kleine historische Beispieldaten
data/sample_fixtures_2026.csv  Beispiel-Fixtures
```

## Nächste Schritte für produktive Nutzung

1. Historische Ergebnisse vollständig laden.
2. Teamnamen normalisieren, z. B. `USA` vs. `United States`.
3. Erwartete Startelf und Verletzungen als eigene Tabellen anbinden.
4. Wetterdaten automatisiert pro Stadion und Anstoßzeit ziehen.
5. Backtesting mit Log Loss und Brier Score durchführen.
6. Modell gegen Wettmarktquoten benchmarken.
