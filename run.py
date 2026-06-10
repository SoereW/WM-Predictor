"""Ein-Befehl-Starter fuer das WM-2026-Dashboard.

    python run.py

Prueft die Abhaengigkeiten (installiert sie bei Bedarf), baut beim ersten Mal
die Datenbank und das Modell aus den **eingecheckten Echtdaten** (offline,
kein Download noetig) und startet anschliessend das Streamlit-Dashboard.

Optionen werden an Streamlit durchgereicht, z. B.:

    python run.py --server.port 8502
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
APP = ROOT / "app.py"

# Modulname -> pip-Paket (nur wo abweichend).
_REQUIRED = ["pandas", "numpy", "sklearn", "streamlit", "plotly", "joblib"]


def _deps_ok() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in _REQUIRED)


def _install_deps() -> None:
    print("• Installiere Abhaengigkeiten (einmalig) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)])


def main() -> None:
    print("=" * 60)
    print(" WM 2026 Predictor – Start")
    print("=" * 60)

    if not _deps_ok():
        try:
            _install_deps()
        except Exception as err:  # noqa: BLE001
            print(f"  Konnte Abhaengigkeiten nicht installieren: {err}")
            print(f"  Bitte manuell: pip install -r {REQUIREMENTS}")
            sys.exit(1)

    sys.path.insert(0, str(ROOT))
    from src.bootstrap import ensure_assets

    ensure_assets(verbose=True)

    print("• Starte Dashboard – im Browser unter http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
