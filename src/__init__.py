"""WM Predictor - Kernpaket.

Hybrides Vorhersagemodell fuer Fussball-Laenderspiele:
Elo-Rating + Dixon-Coles/Poisson-Tor-Modell + ML-Klassifikator,
kalibriert und zeitgewichtet.
"""

__all__ = [
    "database",
    "teams",
    "elo",
    "features",
    "model",
    "simulation",
    "backtest",
    "weather",
    "ingest",
]
