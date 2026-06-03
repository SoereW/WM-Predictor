"""Vergleich der beiden Staerke-Signale als Vorhersage-Systeme.

Es gibt zwei grundsaetzlich verschiedene Quellen fuer "wie stark ist ein
Team":

  (A) SCORE-SYSTEM   - aus echten Spieler-Attributen (EA FC 26) ueber das
                       Bewertungssystem zum Team-Score. Statisch: kennt
                       Talent/Kader, aber keine Resultate/Form.
  (B) HISTORIE-SYSTEM - ML-Modell (Elo + Tor-Modell + Logit) aus ~49k echten
                       Laenderspielen. Kennt Resultate/Form/Heimvorteil, aber
                       nicht den aktuellen Kader.

Dieses Skript misst beide (plus die HYBRID-Kombination, die im Projekt
standardmaessig laeuft) **out-of-sample** an echten Spielen mit bekanntem
Ergebnis und vergleicht Log-Loss, Brier-Score und Trefferquote.

Fairer Aufbau (kein Leakage):
- Das ML-Modell wird einmal auf Spielen VOR ``--cutoff`` trainiert.
- Das Testfenster sind echte Spiele ab ``--cutoff`` bis zum WM-Start,
  ausschliesslich zwischen WM-Nationen (fuer die Score-Ratings vorliegen).
- Elo wird rollierend fortgeschrieben: jede Vorhersage nutzt nur die
  Pre-Match-Ratings, danach wird das Ergebnis eingearbeitet.
- Das Score-System lernt lediglich eine Skala (Score-Differenz -> 1X2) auf
  den Trainingsspielen; die Scores selbst sind die aktuellen Kaderwerte.

    python scripts/compare_systems.py
    python scripts/compare_systems.py --cutoff 2023-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database import connect, read_table
from src.model import WMPredictor
from src.rating.fifa_ingest import load_fifa_as_squads
from src.rating.squad_builder import build_squads
from src.rating.team import rate_all_teams
from src.squad import calibrate_to_elo
from src.wm2026 import GROUP_STAGE_START, all_teams, build_fixtures

DB_PATH = ROOT / "db" / "wm_predictor.sqlite"
FC26_CSV = ROOT / "data" / "fc26_players.csv"
WC_START = GROUP_STAGE_START.strftime("%Y-%m-%d")

# Outcome-Codierung (wie im Modell): 0 = Auswaerts, 1 = Remis, 2 = Heim.
_CLASSES = [0, 1, 2]


def _outcome_code(hs: int, as_: int) -> int:
    return 2 if hs > as_ else (0 if as_ > hs else 1)


def _metrics(y: np.ndarray, P: np.ndarray) -> dict:
    """Log-Loss, Brier (Mehrklassen) und Trefferquote."""
    n = len(y)
    idx = np.arange(n)
    p_true = np.clip(P[idx, y], 1e-12, 1.0)
    log_loss = float(-np.mean(np.log(p_true)))
    onehot = np.zeros_like(P)
    onehot[idx, y] = 1.0
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(P, axis=1) == y))
    return {"log_loss": log_loss, "brier": brier, "accuracy": acc, "n": n}


def _team_scores():
    players = load_fifa_as_squads(FC26_CSV, prefer_long_name=True)
    squads = build_squads(players, all_teams())
    return rate_all_teams(squads)


def _fit_score_model(train: pd.DataFrame, overall: dict) -> LogisticRegression:
    """Score-Differenz (+ Heimfeld) -> 1X2, gelernt auf Trainingsspielen."""
    rows, ys = [], []
    for r in train.itertuples(index=False):
        oh, oa = overall.get(r.home_team), overall.get(r.away_team)
        if oh is None or oa is None:
            continue
        home_field = 0 if int(getattr(r, "neutral", 0) or 0) else 1
        rows.append([oh - oa, home_field])
        ys.append(_outcome_code(int(r.home_score), int(r.away_score)))
    X = np.asarray(rows, dtype=float)
    y = np.asarray(ys, dtype=int)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    return clf


def _score_probs(clf: LogisticRegression, score_diff: float, neutral: int) -> np.ndarray:
    home_field = 0 if neutral else 1
    proba = clf.predict_proba(np.array([[score_diff, home_field]], dtype=float))[0]
    # Spalten nach clf.classes_ ordnen -> [away, draw, home].
    out = np.zeros(3)
    for col, cls in enumerate(clf.classes_):
        out[cls] = proba[col]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score- vs. Historie-System vergleichen")
    parser.add_argument("--cutoff", default="2024-01-01", help="Trainings-/Test-Schnitt (YYYY-MM-DD)")
    args = parser.parse_args()

    con = connect(DB_PATH)
    matches = read_table(con, "matches")
    con.close()
    matches = matches.sort_values("date").reset_index(drop=True)

    team_scores = _team_scores()
    overall = {t: s.overall for t, s in team_scores.items()}

    train = matches[matches["date"] < args.cutoff].reset_index(drop=True)
    scored = set(overall)

    print("=" * 84)
    print(f"VERGLEICH  Score-System (Kader)  vs.  Historie-System (ML)  vs.  Hybrid")
    print(f"Training < {args.cutoff} ({len(train)} Spiele)  |  Test: {args.cutoff} .. {WC_START}")
    print("=" * 84)

    # --- Modelle vorbereiten ----------------------------------------------
    score_clf = _fit_score_model(train, overall)
    base = WMPredictor().fit(train)                 # ML einmal trainiert
    elo_running = base.elo                          # Elo wird rollierend fortgeschrieben

    # Historische Elo (Stand cutoff) -> Score-Kalibrierung fuer das Vergleichsbild.
    hist_elo_at_cutoff = dict(elo_running.ratings)
    calib_cutoff = calibrate_to_elo(overall, hist_elo_at_cutoff)

    # Basisrate (naiv) aus dem Training.
    base_counts = train.apply(lambda r: _outcome_code(int(r.home_score), int(r.away_score)), axis=1)
    base_vec = np.array([(base_counts == c).mean() for c in _CLASSES])
    base_vec = base_vec / base_vec.sum()

    P_score, P_hist, P_hybrid, P_base, Y = [], [], [], [], []
    agree = 0
    n_test = 0

    forward = matches[matches["date"] >= args.cutoff]
    for mrow in forward.itertuples(index=False):
        d = mrow.date
        home, away = mrow.home_team, mrow.away_team
        is_test = d < WC_START and home in scored and away in scored
        if is_test:
            neutral = int(getattr(mrow, "neutral", 0) or 0)
            y = _outcome_code(int(mrow.home_score), int(mrow.away_score))
            hist_form = matches[matches["date"] < d]
            fx = {"home_team": home, "away_team": away, "neutral": neutral}

            # (A) Score-only
            ps = _score_probs(score_clf, overall[home] - overall[away], neutral)

            # (B) Historie-only (kein Kader)
            base.squad_overall, base.squad_calib = {}, None
            ph = base.predict_fixture(fx, hist_form)
            ph_vec = np.array([ph.away_win, ph.draw, ph.home_win])

            # (C) Hybrid (Kader eingekoppelt, mit aktuellem Elo kalibriert)
            base.attach_team_scores(team_scores)
            pc = base.predict_fixture(fx, hist_form)
            pc_vec = np.array([pc.away_win, pc.draw, pc.home_win])

            P_score.append(ps); P_hist.append(ph_vec); P_hybrid.append(pc_vec)
            P_base.append(base_vec); Y.append(y)
            agree += int(np.argmax(ps) == np.argmax(ph_vec))
            n_test += 1

        elo_running.update({"home_team": home, "away_team": away,
                            "home_score": mrow.home_score, "away_score": mrow.away_score,
                            "neutral": getattr(mrow, "neutral", 0), "tournament": getattr(mrow, "tournament", None),
                            "date": d})

    Y = np.asarray(Y, dtype=int)
    A_score, A_hist = np.asarray(P_score), np.asarray(P_hist)
    ensemble = 0.5 * A_score + 0.5 * A_hist            # Prob-Mittel beider Systeme
    res = {
        "Basisrate (naiv)":          _metrics(Y, np.asarray(P_base)),
        "Score-System (Kader)":      _metrics(Y, A_score),
        "Historie-System (ML)":      _metrics(Y, A_hist),
        "Hybrid (Score->Elo->ML)":   _metrics(Y, np.asarray(P_hybrid)),
        "Ensemble O(Score,Historie)": _metrics(Y, ensemble),
    }

    print(f"\nTestspiele: {n_test}\n")
    print(f"{'System':<26}{'Log-Loss':>10}{'Brier':>9}{'Treffer':>10}")
    print("-" * 55)
    for name, m in res.items():
        print(f"{name:<26}{m['log_loss']:>10.4f}{m['brier']:>9.4f}{m['accuracy']*100:>9.1f}%")
    print(f"\nUebereinstimmung der Tipps (Score vs. Historie): {agree}/{n_test} = {agree/max(n_test,1)*100:.0f}%")
    print("(niedriger Log-Loss/Brier = besser; Basisrate ~1.05 ist die naive Referenz)")

    _print_strength_gap(overall, hist_elo_at_cutoff, calib_cutoff)
    _print_disagreements(score_clf, base, team_scores, overall, matches, args.cutoff)


def _print_strength_gap(overall, hist_elo, calib) -> None:
    """Wo weicht die Kader-Staerke am staerksten von der historischen Elo ab?"""
    if calib is None:
        return
    a, b = calib
    rows = []
    for t, ovr in overall.items():
        if t in hist_elo:
            score_elo = a + b * ovr
            rows.append((t, score_elo - hist_elo[t]))
    rows.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 84)
    print("KADER-STAERKE vs. HISTORISCHE ELO  (Score-Elo minus Historie-Elo, Stand cutoff)")
    print("=" * 84)
    print("  Kader STARK, Historie schwach (Talent noch nicht in Resultaten):")
    for t, gap in rows[:6]:
        print(f"    {t:<20} {gap:+6.0f} Elo")
    print("  Historie STARK, Kader schwaecher (Resultate/Erfahrung > Kaderwerte):")
    for t, gap in rows[-6:][::-1]:
        print(f"    {t:<20} {gap:+6.0f} Elo")


def _print_disagreements(score_clf, base, team_scores, overall, matches, cutoff) -> None:
    """WM-Spiele, bei denen Score- und Historie-System am staerksten abweichen."""
    base.attach_team_scores(team_scores)  # Hybrid-Zustand egal, wir nutzen beide getrennt
    fixtures = build_fixtures()
    hist_form = matches[matches["date"] < WC_START]
    rows = []
    for fx in fixtures.itertuples(index=False):
        home, away, neutral = fx.home_team, fx.away_team, int(fx.neutral)
        if home not in overall or away not in overall:
            continue
        ps = _score_probs(score_clf, overall[home] - overall[away], neutral)
        base.squad_overall, base.squad_calib = {}, None
        ph = base.predict_fixture({"home_team": home, "away_team": away, "neutral": neutral}, hist_form)
        diff = abs(ps[2] - ph.home_win)
        rows.append((diff, home, away, ps[2], ph.home_win))
    rows.sort(reverse=True)
    print("\n" + "=" * 84)
    print("GROESSTE UNEINIGKEIT auf WM-Spielen  (P(Heimsieg): Score vs. Historie)")
    print("=" * 84)
    for diff, home, away, p_s, p_h in rows[:10]:
        print(f"  {home:<16} vs {away:<16}  Score {p_s*100:4.0f}%  |  Historie {p_h*100:4.0f}%  (Δ {diff*100:.0f})")


if __name__ == "__main__":
    main()
