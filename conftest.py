"""Rend la racine du projet importable par pytest.

Sans cela, `src.features` — référencé par le pipeline sérialisé — resterait
introuvable et le rechargement joblib échouerait.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))