"""Tests du pipeline sérialisé et de l'API.

Lancement :
    pytest -v

Les tests portent sur le modèle **rechargé depuis le disque**, c'est-à-dire
exactement l'objet que sert l'API — et non sur un pipeline reconstruit en
mémoire, qui pourrait masquer un problème de sérialisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = RACINE / "model" / "pipeline_churn.joblib"
CHEMIN_META = RACINE / "model" / "metadata.json"

pytestmark = pytest.mark.skipif(
    not CHEMIN_MODELE.exists(),
    reason="Modèle absent — exécuter `python src/train.py` au préalable")


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(CHEMIN_MODELE)


@pytest.fixture(scope="module")
def metadonnees():
    return json.loads(CHEMIN_META.read_text(encoding="utf-8"))


@pytest.fixture
def client_type() -> dict:
    """Client à haut risque : mensuel, fibre, ancienneté faible."""
    return {
        "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
        "TotalCharges": 171.0,
    }


# --------------------------------------------------------------------------
# 1. Forme de la sortie
# --------------------------------------------------------------------------
def test_sortie_a_la_bonne_forme(pipeline, client_type):
    """n lignes en entrée → matrice (n, 2) en sortie, une colonne par classe."""
    lot = pd.DataFrame([client_type] * 7)
    proba = pipeline.predict_proba(lot)
    assert proba.shape == (7, 2)
    assert pipeline.predict(lot).shape == (7,)


# --------------------------------------------------------------------------
# 2. Probabilités dans [0, 1]
# --------------------------------------------------------------------------
def test_probabilites_dans_intervalle_unitaire(pipeline, client_type):
    """Contrainte structurelle : les deux classes somment à 1 et restent bornées."""
    variantes = []
    for tenure in (0, 1, 12, 72):
        for contrat in ("Month-to-month", "One year", "Two year"):
            variantes.append({**client_type, "tenure": tenure, "Contract": contrat})
    proba = pipeline.predict_proba(pd.DataFrame(variantes))
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# 3. Gestion des valeurs manquantes
# --------------------------------------------------------------------------
def test_valeurs_manquantes_gerees(pipeline, client_type):
    """Un nouveau client a TotalCharges absent : le pipeline doit l'imputer,
    pas lever une exception. C'est le cas des 11 lignes repérées en M1."""
    nouveau = {**client_type, "tenure": 0, "TotalCharges": np.nan}
    proba = pipeline.predict_proba(pd.DataFrame([nouveau]))[0, 1]
    assert np.isfinite(proba) and 0.0 <= proba <= 1.0


# --------------------------------------------------------------------------
# 4. Features attendues présentes
# --------------------------------------------------------------------------
def test_colonne_manquante_rejetee(pipeline, metadonnees, client_type):
    """Une colonne absente doit provoquer une erreur explicite. Un pipeline
    qui prédirait silencieusement sur des données incomplètes est pire
    qu'un pipeline qui échoue."""
    attendues = set(metadonnees["colonnes_attendues"])
    assert attendues == set(client_type), "Le client de test ne couvre pas le schéma"

    ampute = {k: v for k, v in client_type.items() if k != "Contract"}
    with pytest.raises(Exception):
        pipeline.predict_proba(pd.DataFrame([ampute]))


# --------------------------------------------------------------------------
# 5. Performance sur un jeu de référence
# --------------------------------------------------------------------------
def test_performance_ne_regresse_pas(pipeline, metadonnees):
    """Garde-fou anti-régression : le modèle rechargé doit reproduire les
    performances enregistrées lors de l'entraînement, à tolérance près."""
    from src.features import RANDOM_STATE, charger_donnees, cout_metier
    from sklearn.model_selection import train_test_split

    X, y = charger_donnees(str(RACINE / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    proba = pipeline.predict_proba(X_test)[:, 1]
    cout = cout_metier(y_test, proba)
    attendu = metadonnees["performances_test"]["cout_metier_eur_par_client"]

    assert cout == pytest.approx(attendu, abs=0.5), (
        f"Coût {cout:.2f} € contre {attendu} € attendu — le modèle a changé")
    assert cout < 65, "Cible du cadrage (M0) non atteinte"


# --------------------------------------------------------------------------
# 6. Déterminisme de la sérialisation
# --------------------------------------------------------------------------
def test_predictions_identiques_apres_rechargement(pipeline, client_type):
    """Exigence explicite de l'énoncé : après rechargement, les prédictions
    doivent être *strictement* identiques, pas seulement proches."""
    lot = pd.DataFrame([client_type] * 5)
    autre_instance = joblib.load(CHEMIN_MODELE)
    assert np.array_equal(pipeline.predict_proba(lot),
                          autre_instance.predict_proba(lot))


# --------------------------------------------------------------------------
# 7. Cohérence métier
# --------------------------------------------------------------------------
def test_engagement_reduit_le_risque(pipeline, client_type):
    """Un contrat de deux ans doit réduire la probabilité de churn par rapport
    au mensuel, toutes choses égales par ailleurs. Ce test échouerait si un
    encodage se désalignait silencieusement."""
    mensuel = {**client_type, "Contract": "Month-to-month"}
    engage = {**client_type, "Contract": "Two year"}
    p = pipeline.predict_proba(pd.DataFrame([mensuel, engage]))[:, 1]
    assert p[0] > p[1], "L'engagement contractuel devrait réduire le risque"


# --------------------------------------------------------------------------
# 8. Endpoints de l'API
# --------------------------------------------------------------------------
def test_api_endpoints(client_type):
    from api.main import app

    with TestClient(app) as testeur:
        sante = testeur.get("/health")
        assert sante.status_code == 200 and sante.json()["statut"] == "ok"

        infos = testeur.get("/model-info")
        assert infos.status_code == 200
        assert "seuil_decision" in infos.json()

        reponse = testeur.post("/predict", json=client_type)
        assert reponse.status_code == 200
        corps = reponse.json()
        assert 0.0 <= corps["probabilite_churn"] <= 1.0
        assert isinstance(corps["cibler"], bool)

        # Une modalité inconnue doit être rejetée par la validation Pydantic
        invalide = {**client_type, "Contract": "Trois ans"}
        assert testeur.post("/predict", json=invalide).status_code == 422