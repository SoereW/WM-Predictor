"""Hybrides Vorhersagemodell.

Drei Bausteine, kalibriert kombiniert:

1. **World-Football-Elo** als robuste Staerke-Baseline.
2. **Dixon-Coles/Poisson-Tor-Modell** -> erwartete Tore + konsistente
   1X2-/Korrektergebnis-Wahrscheinlichkeiten.
3. **Multinomialer Logit-Klassifikator** auf den Pre-Match-Features.

Tor-Modell und Klassifikator werden gewichtet gemischt; Mischgewicht und
die Dixon-Coles-Korrektur ``rho`` werden auf einem zeitlich abgetrennten
Validierungsfenster per Grid-Search auf minimalen Log-Loss gewaehlt. Das
Ergebnis ist ein *kalibrierter* Wahrscheinlichkeitsvektor statt einer
nackten Tipp-Klasse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .elo import HOME_ADVANTAGE, EloModel
from .features import (
    CLASSIFIER_FEATURES,
    GOALS_FEATURES,
    build_features,
    classifier_matrix,
    stacked_goals_frame,
    team_state,
)

MAX_GOALS = 10
ADJ_ELO_SCALE = 150.0  # 1.0 Adjustment ~ 150 Elo-Punkte
# Anteil, mit dem die Kaderstaerke das effektive Elo in ihre Richtung zieht.
# Moderat gewaehlt: Elo bleibt Basis, der Kader korrigiert dort, wo Historie
# und aktuelle Kaderqualitaet auseinanderlaufen.
SQUAD_PULL = 0.35
RHO_GRID = [0.0, -0.04, -0.08, -0.12, -0.16]
WEIGHT_GRID = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
_LOG_FACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, MAX_GOALS + 1)))])


@dataclass
class Prediction:
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float
    top_factors: List[Tuple[str, float]] = field(default_factory=list)


def _poisson_pmf(lams, kmax: int = MAX_GOALS) -> np.ndarray:
    """Poisson-Wahrscheinlichkeiten P(X=k) fuer k=0..kmax, vektorisiert."""
    lams = np.clip(np.asarray(lams, dtype=float), 1e-6, None)[:, None]
    ks = np.arange(kmax + 1)[None, :]
    log_pmf = -lams + ks * np.log(lams) - _LOG_FACT[None, :]
    return np.exp(log_pmf)


def _matrix_probs(lh, la, rho: float, kmax: int = MAX_GOALS) -> np.ndarray:
    """1X2-Wahrscheinlichkeiten aus dem Dixon-Coles-Score-Gitter.

    Liefert ein Array der Form (n, 3): [Auswaertssieg, Remis, Heimsieg].
    """
    lh = np.atleast_1d(np.asarray(lh, dtype=float))
    la = np.atleast_1d(np.asarray(la, dtype=float))
    pmf_h = _poisson_pmf(lh, kmax)
    pmf_a = _poisson_pmf(la, kmax)
    m = pmf_h[:, :, None] * pmf_a[:, None, :]  # (n, K+1, K+1) -> [home, away]

    # Dixon-Coles-Korrektur fuer die niedrigen Ergebnisse.
    m[:, 0, 0] *= 1.0 - lh * la * rho
    m[:, 0, 1] *= 1.0 + lh * rho
    m[:, 1, 0] *= 1.0 + la * rho
    m[:, 1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    m /= m.sum(axis=(1, 2), keepdims=True)

    idx = np.arange(kmax + 1)
    home_mask = idx[:, None] > idx[None, :]
    away_mask = idx[:, None] < idx[None, :]
    draw_mask = idx[:, None] == idx[None, :]
    p_home = m[:, home_mask].sum(axis=1)
    p_away = m[:, away_mask].sum(axis=1)
    p_draw = m[:, draw_mask].sum(axis=1)
    probs = np.stack([p_away, p_draw, p_home], axis=1)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _log_loss(y: np.ndarray, P: np.ndarray) -> float:
    P = np.clip(P, 1e-12, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def _brier(y: np.ndarray, P: np.ndarray) -> float:
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def _accuracy(y: np.ndarray, P: np.ndarray) -> float:
    return float(np.mean(np.argmax(P, axis=1) == y))


def _metrics(y: np.ndarray, P: np.ndarray) -> Dict[str, float]:
    return {
        "log_loss": round(_log_loss(y, P), 4),
        "brier": round(_brier(y, P), 4),
        "accuracy": round(_accuracy(y, P), 4),
    }


class WMPredictor:
    """Trainierbares hybrides Vorhersagemodell."""

    def __init__(self, max_goals: int = MAX_GOALS, squad_pull: float = SQUAD_PULL):
        self.max_goals = max_goals
        self.squad_pull = squad_pull
        self.elo: EloModel | None = None
        self.goals_model: Pipeline | None = None
        self.clf: Pipeline | None = None
        self.blend_weight: float = 0.5
        self.rho: float = -0.08
        # Optionale Kaderschicht (wird erst zum Vorhersagezeitpunkt genutzt).
        self.squad_overall: Dict[str, float] = {}
        self.squad_calib: Tuple[float, float] | None = None
        # Vertrauen je Team in die Kaderstaerke (0..1, aus Datenabdeckung).
        # Duenn abgedeckte Nationen ziehen den Score nur abgeschwaecht.
        self.squad_confidence: Dict[str, float] = {}
        self.training_summary: Dict = {}

    # ----------------------------------------------------------------- fit
    def _fit_goals(self, feat_df: pd.DataFrame) -> Pipeline:
        frame = stacked_goals_frame(feat_df)
        model = Pipeline(
            [("scale", StandardScaler()), ("poisson", PoissonRegressor(alpha=1e-3, max_iter=400))]
        )
        model.fit(frame[GOALS_FEATURES], frame["goals"], poisson__sample_weight=frame["weight"])
        return model

    def _fit_clf(self, feat_df: pd.DataFrame) -> Pipeline:
        X = classifier_matrix(feat_df)
        model = Pipeline(
            [("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=3000))]
        )
        model.fit(X, feat_df["result"].astype(int), logit__sample_weight=feat_df["time_weight"])
        return model

    def _goals_probs(self, goals_model: Pipeline, feat_df: pd.DataFrame, rho: float) -> np.ndarray:
        frame = stacked_goals_frame(feat_df)
        n = len(feat_df)
        lam = np.clip(goals_model.predict(frame[GOALS_FEATURES]), 0.05, 8.0)
        return _matrix_probs(lam[:n], lam[n:], rho, self.max_goals)

    def fit(self, matches: pd.DataFrame) -> "WMPredictor":
        feat_df, elo = build_features(matches)
        self.elo = elo

        # Zeitlich abgetrennter Holdout zum Tunen von rho und Mischgewicht.
        feat_df = feat_df.sort_values("date").reset_index(drop=True)
        split = int(len(feat_df) * 0.8)
        train, val = feat_df.iloc[:split], feat_df.iloc[split:]
        summary: Dict = {}

        if len(val) >= 200 and len(train) >= 200:
            g_tmp = self._fit_goals(train)
            c_tmp = self._fit_clf(train)
            y_val = val["result"].astype(int).to_numpy()
            clf_val = c_tmp.predict_proba(classifier_matrix(val))

            goals_val_cache = {rho: self._goals_probs(g_tmp, val, rho) for rho in RHO_GRID}
            best = (np.inf, -0.08, 0.5)
            for rho in RHO_GRID:
                gp = goals_val_cache[rho]
                for w in WEIGHT_GRID:
                    blended = w * gp + (1 - w) * clf_val
                    blended /= blended.sum(axis=1, keepdims=True)
                    ll = _log_loss(y_val, blended)
                    if ll < best[0]:
                        best = (ll, rho, w)
            _, self.rho, self.blend_weight = best

            gp_best = goals_val_cache[self.rho]
            blended_best = self.blend_weight * gp_best + (1 - self.blend_weight) * clf_val
            blended_best /= blended_best.sum(axis=1, keepdims=True)

            base_rates = train["result"].astype(int).value_counts(normalize=True)
            base_vec = np.array(
                [base_rates.get(0, 1 / 3), base_rates.get(1, 1 / 3), base_rates.get(2, 1 / 3)]
            )
            base_vec /= base_vec.sum()
            base_P = np.tile(base_vec, (len(val), 1))

            elo_only = Pipeline(
                [("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=2000))]
            )
            elo_only.fit(
                train[["elo_diff"]], train["result"].astype(int),
                logit__sample_weight=train["time_weight"],
            )
            elo_P = elo_only.predict_proba(val[["elo_diff"]])

            summary["validation"] = {
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "baseline_base_rate": _metrics(y_val, base_P),
                "baseline_elo_only": _metrics(y_val, elo_P),
                "goals_model": _metrics(y_val, gp_best),
                "classifier": _metrics(y_val, clf_val),
                "blended_model": _metrics(y_val, blended_best),
            }

        # Finales Training auf allen Daten mit den gewaehlten Hyperparametern.
        self.goals_model = self._fit_goals(feat_df)
        self.clf = self._fit_clf(feat_df)

        summary.update(
            {
                "n_matches": int(len(feat_df)),
                "date_from": str(feat_df["date"].min()),
                "date_to": str(feat_df["date"].max()),
                "n_teams": int(len(self.elo.ratings)),
                "home_advantage_elo": HOME_ADVANTAGE,
                "dixon_coles_rho": round(self.rho, 3),
                "blend_weight_goals_model": round(self.blend_weight, 3),
                "top_teams_by_elo": [
                    [t, round(r, 1)]
                    for t, r in sorted(self.elo.ratings.items(), key=lambda kv: kv[1], reverse=True)[:10]
                ],
            }
        )
        self.training_summary = summary
        return self

    # --------------------------------------------------------------- squad
    def attach_squads(self, squads: pd.DataFrame) -> "WMPredictor":
        """Bindet Kaderdaten an und kalibriert Teamstaerke auf die Elo-Skala.

        Greift bewusst erst zur *Vorhersage* - die trainierten ML-Modelle
        sehen die Kaderstaerke nie und koennen sie daher nicht overfitten.
        """
        from .squad import calibrate_to_elo, team_overall_map

        if squads is None or len(squads) == 0:
            self.squad_overall = {}
            self.squad_calib = None
            return self

        self.squad_overall = team_overall_map(squads)
        self.squad_calib = calibrate_to_elo(self.squad_overall, self.elo.ratings)
        summary = dict(self.training_summary)
        summary["squad"] = {
            "n_teams_with_squad": int(len(self.squad_overall)),
            "calibrated": self.squad_calib is not None,
            "squad_pull": round(self.squad_pull, 2),
        }
        if self.squad_calib is not None:
            a, b = self.squad_calib
            summary["squad"]["overall_to_elo"] = {"intercept": round(a, 1), "slope": round(b, 1)}
        self.training_summary = summary
        return self

    def attach_team_scores(self, team_scores: Dict, coverage: Dict | None = None) -> "WMPredictor":
        """Bindet das volle Bewertungssystem (Spieler+Chemie+Trainer) an.

        Reichhaltigere Alternative zu ``attach_squads``: statt des reinen
        Spielerdurchschnitts wird der Team-Gesamtscore aus ``src.rating``
        (beste Elf, Tiefe, Chemie, Trainer) als Staerkequelle genutzt und
        wie gehabt auf die Elo-Skala kalibriert. Greift ebenfalls erst zur
        Vorhersage - die trainierten ML-Modelle sehen ihn nie.

        ``team_scores`` ist ein Dict ``{team_name: objekt_mit_.overall}``
        (z. B. die Ausgabe von ``rate_all_teams``).

        ``coverage`` ist optional ``{team_name: konfidenz 0..1}`` aus der
        Datenabdeckung (z. B. Anzahl verfuegbarer Spieler / voller Kader).
        Bei duenner Abdeckung (wenige Spieler im Datensatz) wird der
        Kader-Score nur abgeschwaecht eingekoppelt - das Modell vertraut dann
        staerker der historischen Elo. Ohne Angabe gilt volles Vertrauen (1.0).
        """
        from .squad import calibrate_to_elo

        if not team_scores:
            self.squad_overall = {}
            self.squad_calib = None
            self.squad_confidence = {}
            return self

        self.squad_overall = {
            t: float(getattr(s, "overall", s)) for t, s in team_scores.items()
        }
        self.squad_confidence = {
            t: float(np.clip(c, 0.0, 1.0)) for t, c in (coverage or {}).items()
        }
        self.squad_calib = calibrate_to_elo(self.squad_overall, self.elo.ratings)
        summary = dict(self.training_summary)
        summary["squad"] = {
            "source": "rating_system (Spieler+Chemie+Trainer)",
            "n_teams_with_squad": int(len(self.squad_overall)),
            "calibrated": self.squad_calib is not None,
            "squad_pull": round(self.squad_pull, 2),
            "coverage_confidence": bool(self.squad_confidence),
        }
        if self.squad_calib is not None:
            a, b = self.squad_calib
            summary["squad"]["overall_to_elo"] = {"intercept": round(a, 1), "slope": round(b, 1)}
        self.training_summary = summary
        return self

    def effective_elo(self, team: str, elo_rating: float) -> float:
        """Elo nach moderater Korrektur durch die Kaderstaerke.

        Ohne Kaderdaten (oder ohne Kalibrierung) bleibt das reine Elo
        unveraendert. Liegt eine kalibrierte Kaderstaerke vor, wird das
        Rating mit ``squad_pull`` in deren Richtung gezogen.
        """
        if not self.squad_calib:
            return elo_rating
        from .squad import squad_to_elo

        ovr = self.squad_overall.get(team)
        if ovr is None or ovr != ovr:  # fehlend oder NaN
            return elo_rating
        squad_elo = squad_to_elo(ovr, self.squad_calib)
        # Pull je Team mit der Datenabdeckung skalieren: duenn abgedeckte
        # Kader (wenige Spieler im Datensatz) ziehen die Elo nur abgeschwaecht.
        conf = getattr(self, "squad_confidence", {}).get(team, 1.0)
        pull = self.squad_pull * conf
        return (1.0 - pull) * elo_rating + pull * squad_elo

    # ------------------------------------------------------------- predict
    def _core(self, home_elo, away_elo, hs, as_, neutral: int, adj_total: float = 0.0,
              extra_elo: float = 0.0):
        """Gemeinsamer Vorhersagekern fuer ein Matchup.

        ``extra_elo`` sind zusaetzliche Elo-Punkte (Heimsicht) aus Kontext-
        (Reise/Pause/Hoehe/Klima) und Taktik-Faktoren.
        """
        adj_elo = adj_total * ADJ_ELO_SCALE + extra_elo
        hfa = 0.0 if neutral else HOME_ADVANTAGE
        elo_diff_eff = (home_elo - away_elo) + adj_elo + hfa

        clf_row = pd.DataFrame(
            [
                {
                    "elo_diff": elo_diff_eff,
                    "form_pts_diff": hs.form_pts - as_.form_pts,
                    "home_form_gf": hs.form_gf,
                    "home_form_ga": hs.form_ga,
                    "away_form_gf": as_.form_gf,
                    "away_form_ga": as_.form_ga,
                    "rest_diff": 0.0,
                    "neutral": int(neutral),
                    "sos_diff": (hs.elo - as_.elo) / 100.0,
                    "form_vs_exp_diff": 0.0,
                }
            ]
        )[CLASSIFIER_FEATURES]
        clf_probs = self.clf.predict_proba(clf_row)[0]

        elo_adv = (home_elo - away_elo + adj_elo) / 100.0
        goals_rows = pd.DataFrame(
            [
                {
                    "is_home": 0.0 if neutral else 1.0,
                    "elo_adv": elo_adv,
                    "own_attack": hs.form_gf,
                    "opp_defense": as_.form_ga,
                    "own_form": hs.form_pts,
                },
                {
                    "is_home": 0.0,
                    "elo_adv": -elo_adv,
                    "own_attack": as_.form_gf,
                    "opp_defense": hs.form_ga,
                    "own_form": as_.form_pts,
                },
            ]
        )[GOALS_FEATURES]
        lam = np.clip(self.goals_model.predict(goals_rows), 0.05, 8.0)
        lh, la = float(lam[0]), float(lam[1])
        goals_probs = _matrix_probs([lh], [la], self.rho, self.max_goals)[0]

        w = self.blend_weight
        probs = w * goals_probs + (1 - w) * clf_probs
        probs = probs / probs.sum()
        return probs, lh, la, elo_diff_eff

    def predict_fixture(self, fixture: Dict, matches: pd.DataFrame, adjustments: Dict | None = None,
                        context_elo: float = 0.0, tactics_elo: float = 0.0) -> Prediction:
        from .teams import normalize_team

        home = normalize_team(fixture.get("home_team"))
        away = normalize_team(fixture.get("away_team"))
        neutral = int(fixture.get("neutral", 1) or 0)
        adjustments = adjustments or {}
        adj_total = float(
            adjustments.get("player_delta", 0.0)
            + adjustments.get("injury_delta", 0.0)
            + adjustments.get("rest_delta", 0.0)
            + adjustments.get("weather_delta", 0.0)
        )
        extra_elo = float(context_elo) + float(tactics_elo)

        hs = team_state(matches, self.elo, home)
        as_ = team_state(matches, self.elo, away)
        home_elo = self.effective_elo(home, hs.elo)
        away_elo = self.effective_elo(away, as_.elo)
        probs, lh, la, elo_diff_eff = self._core(home_elo, away_elo, hs, as_, neutral, adj_total, extra_elo)

        top_factors = [
            (f"Elo {home}", round(hs.elo, 0)),
            (f"Elo {away}", round(as_.elo, 0)),
        ]
        if self.squad_calib:
            top_factors += [
                (f"Elo+Kader {home}", round(home_elo, 0)),
                (f"Elo+Kader {away}", round(away_elo, 0)),
            ]
            ovr_h, ovr_a = self.squad_overall.get(home), self.squad_overall.get(away)
            if ovr_h is not None and ovr_h == ovr_h:
                top_factors.append((f"Kaderstaerke {home}", round(ovr_h, 1)))
            if ovr_a is not None and ovr_a == ovr_a:
                top_factors.append((f"Kaderstaerke {away}", round(ovr_a, 1)))
        if abs(context_elo) > 0.05:
            top_factors.append(("Kontext (Reise/Pause/Hoehe/Klima) Elo", round(context_elo, 1)))
        if abs(tactics_elo) > 0.05:
            top_factors.append(("Taktik-Matchup Elo", round(tactics_elo, 1)))
        top_factors += [
            ("Elo-Differenz (mit Heimvorteil/Adjust)", round(elo_diff_eff, 1)),
            (f"Form-Punkte {home}", round(hs.form_pts, 2)),
            (f"Form-Punkte {away}", round(as_.form_pts, 2)),
            (f"Erw. Tore {home}", round(lh, 2)),
            (f"Erw. Tore {away}", round(la, 2)),
            ("Neutraler Platz", "ja" if neutral else "nein"),
            ("Kontext-Adjustment gesamt", round(adj_total, 2)),
        ]
        return Prediction(
            home_win=float(probs[2]),
            draw=float(probs[1]),
            away_win=float(probs[0]),
            expected_home_goals=lh,
            expected_away_goals=la,
            top_factors=top_factors,
        )

    def _score_grid(self, lh: float, la: float) -> np.ndarray:
        """Normalisiertes Dixon-Coles-Korrektergebnis-Gitter aus erwarteten Toren."""
        pmf_h = _poisson_pmf([lh], self.max_goals)[0]
        pmf_a = _poisson_pmf([la], self.max_goals)[0]
        m = np.outer(pmf_h, pmf_a)
        m[0, 0] *= 1.0 - lh * la * self.rho
        m[0, 1] *= 1.0 + lh * self.rho
        m[1, 0] *= 1.0 + la * self.rho
        m[1, 1] *= 1.0 - self.rho
        m = np.clip(m, 0.0, None)
        return m / m.sum()

    def score_matrix(self, fixture: Dict, matches: pd.DataFrame, adjustments: Dict | None = None) -> np.ndarray:
        """Normalisiertes Korrektergebnis-Gitter (fuer Monte-Carlo-Simulation)."""
        from .teams import normalize_team

        home = normalize_team(fixture.get("home_team"))
        away = normalize_team(fixture.get("away_team"))
        neutral = int(fixture.get("neutral", 1) or 0)
        adjustments = adjustments or {}
        adj_total = float(
            sum(adjustments.get(k, 0.0) for k in ("player_delta", "injury_delta", "rest_delta", "weather_delta"))
        )

        hs = team_state(matches, self.elo, home)
        as_ = team_state(matches, self.elo, away)
        home_elo = self.effective_elo(home, hs.elo)
        away_elo = self.effective_elo(away, as_.elo)
        _, lh, la, _ = self._core(home_elo, away_elo, hs, as_, neutral, adj_total)
        return self._score_grid(lh, la)

    # ---------------------------------------------- vorbereitete Team-Zustaende
    def _state(self, team: str, states: Dict):
        """Team-Zustand aus dem Cache holen (oder einen neutralen Fallback bauen).

        Erlaubt es, alle paarweisen Vorhersagen eines Turniers aus einmalig
        vorberechneten ``TeamState``-Objekten abzuleiten - ohne die Historie je
        Spiel erneut zu durchsuchen (das macht die Turnier-Simulation schnell).
        """
        st = states.get(team)
        if st is not None:
            return st
        elo = self.elo.current_rating(team) if self.elo is not None else 1500.0
        from .features import TeamState

        return TeamState(elo=elo, form_gf=1.2, form_ga=1.2, form_pts=1.2)

    def matchup_from_states(self, home: str, away: str, neutral, states: Dict, extra_elo: float = 0.0):
        """1X2 + erwartete Tore eines Duells aus vorbereiteten Team-Zustaenden.

        Rueckgabe: ``(p_home, p_draw, p_away, xg_home, xg_away)``.
        """
        hs, as_ = self._state(home, states), self._state(away, states)
        home_elo = self.effective_elo(home, hs.elo)
        away_elo = self.effective_elo(away, as_.elo)
        probs, lh, la, _ = self._core(home_elo, away_elo, hs, as_, int(neutral or 0), 0.0, float(extra_elo))
        return float(probs[2]), float(probs[1]), float(probs[0]), float(lh), float(la)

    def score_grid_from_states(self, home: str, away: str, neutral, states: Dict, extra_elo: float = 0.0) -> np.ndarray:
        """Korrektergebnis-Gitter eines Duells aus vorbereiteten Team-Zustaenden."""
        hs, as_ = self._state(home, states), self._state(away, states)
        home_elo = self.effective_elo(home, hs.elo)
        away_elo = self.effective_elo(away, as_.elo)
        _, lh, la, _ = self._core(home_elo, away_elo, hs, as_, int(neutral or 0), 0.0, float(extra_elo))
        return self._score_grid(lh, la)

    # ---------------------------------------------------------------- io
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "WMPredictor":
        return joblib.load(Path(path))
