"""Out-of-Sample-Backtesting.

Belegt die Vorhersagequalitaet objektiv: trainiert auf der Vergangenheit,
prognostiziert die Zukunft und misst Log-Loss, Brier-Score und
Trefferquote - jeweils gegen eine Basisrate- und eine Elo-only-Baseline.
Zusaetzlich eine Kalibrierungstabelle (vorhergesagte vs. tatsaechliche
Favoritenquote).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features, classifier_matrix
from .model import WMPredictor, _metrics


def _predict_blended(predictor: WMPredictor, feat_df: pd.DataFrame) -> np.ndarray:
    clf_probs = predictor.clf.predict_proba(classifier_matrix(feat_df))
    goals_probs = predictor._goals_probs(predictor.goals_model, feat_df, predictor.rho)
    w = predictor.blend_weight
    blended = w * goals_probs + (1 - w) * clf_probs
    return blended / blended.sum(axis=1, keepdims=True)


def _calibration(y: np.ndarray, P: np.ndarray, bins: int = 10) -> list:
    """Kalibrierung der hoechsten vorhergesagten Klasse (Favorit)."""
    pred_cls = np.argmax(P, axis=1)
    conf = P[np.arange(len(y)), pred_cls]
    correct = (pred_cls == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    table = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf >= lo) & (conf < hi if b < bins - 1 else conf <= hi)
        if mask.sum() == 0:
            continue
        table.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "predicted": round(float(conf[mask].mean()), 3),
                "actual": round(float(correct[mask].mean()), 3),
            }
        )
    return table


def run_backtest(matches: pd.DataFrame, train_frac: float = 0.85) -> Dict:
    """Trainiert auf den ersten ``train_frac`` der Historie, testet auf dem Rest."""
    matches = matches.sort_values("date").reset_index(drop=True)
    split = int(len(matches) * train_frac)
    train_matches = matches.iloc[:split]

    predictor = WMPredictor().fit(train_matches)

    # Features fuer das Testfenster mit dem auf Train gefitteten Elo bauen
    # (Pre-Match-Ratings/Form weiter fortgeschrieben, kein Leakage).
    feat_all, _ = build_features(matches, elo_model=predictor.elo)
    feat_test = feat_all.iloc[split:].reset_index(drop=True)
    feat_train = feat_all.iloc[:split]
    y = feat_test["result"].astype(int).to_numpy()

    blended = _predict_blended(predictor, feat_test)

    base_rates = feat_train["result"].astype(int).value_counts(normalize=True)
    base_vec = np.array([base_rates.get(0, 1 / 3), base_rates.get(1, 1 / 3), base_rates.get(2, 1 / 3)])
    base_vec /= base_vec.sum()
    base_P = np.tile(base_vec, (len(y), 1))

    elo_only = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=2000))])
    elo_only.fit(
        feat_train[["elo_diff"]], feat_train["result"].astype(int),
        logit__sample_weight=feat_train["time_weight"],
    )
    elo_P = elo_only.predict_proba(feat_test[["elo_diff"]])

    return {
        "n_train": int(len(train_matches)),
        "n_test": int(len(y)),
        "date_test_from": str(feat_test["date"].min()),
        "date_test_to": str(feat_test["date"].max()),
        "metrics": {
            "baseline_base_rate": _metrics(y, base_P),
            "baseline_elo_only": _metrics(y, elo_P),
            "blended_model": _metrics(y, blended),
        },
        "calibration_blended": _calibration(y, blended),
    }
