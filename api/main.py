"""API de scoring du churn client.

Lancement :
    uvicorn api.main:app --reload

Documentation interactive : http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RACINE = Path(__file__).resolve().parent.parent
# Le pipeline sérialisé référence src.features : la racine doit être importable.
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))
CHEMIN_MODELE = RACINE / "model" / "pipeline_churn.joblib"
CHEMIN_META = RACINE / "model" / "metadata.json"

app = FastAPI(
    title="API de prédiction du churn client",
    description="Score le risque de résiliation d'un client télécom "
                "et recommande son ciblage par la campagne de rétention.",
    version="1.0.0",
)

# Chargé une seule fois au démarrage : recharger le modèle à chaque requête
# multiplierait la latence par cent sans aucun bénéfice.
try:
    _pipeline = joblib.load(CHEMIN_MODELE)
    _meta = json.loads(CHEMIN_META.read_text(encoding="utf-8"))
except FileNotFoundError:
    _pipeline, _meta = None, None


class Client(BaseModel):
    """Un client à scorer. Les contraintes sont validées par Pydantic
    avant d'atteindre le modèle : une requête malformée reçoit un 422
    explicite plutôt qu'une erreur interne."""

    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120, description="Ancienneté en mois")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Bank transfer (automatic)", "Credit card (automatic)",
                           "Electronic check", "Mailed check"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0,
                                       description="Null accepté : nouveau client non encore facturé")

    model_config = {"json_schema_extra": {"examples": [{
        "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
        "TotalCharges": 171.0}]}}


class Prediction(BaseModel):
    probabilite_churn: float
    cibler: bool
    seuil_applique: float
    version_modele: str


@app.get("/health", summary="Statut du service")
def health() -> dict:
    """Indique si le modèle est chargé et prêt à répondre."""
    if _pipeline is None:
        raise HTTPException(status_code=503,
                            detail="Modèle indisponible — exécuter d'abord src/train.py")
    return {"statut": "ok", "modele_charge": True, "version": _meta["version"]}


@app.post("/predict", response_model=Prediction, summary="Score un client")
def predict(client: Client) -> Prediction:
    """Retourne la probabilité de résiliation et la décision de ciblage.

    Le seuil appliqué (0,1026) n'est pas 0,5 : il découle de la matrice de
    coûts métier, un faux négatif coûtant environ neuf fois un faux positif.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible")

    ligne = pd.DataFrame([client.model_dump()])
    seuil = _meta["seuil_decision"]
    proba = float(_pipeline.predict_proba(ligne)[0, 1])

    return Prediction(probabilite_churn=round(proba, 4),
                      cibler=bool(proba >= seuil),
                      seuil_applique=seuil,
                      version_modele=_meta["version"])


@app.get("/model-info", summary="Métadonnées du modèle")
def model_info() -> dict:
    """Features attendues, seuil, coûts métier et performances de validation."""
    if _meta is None:
        raise HTTPException(status_code=503, detail="Métadonnées indisponibles")
    return {
        "version": _meta["version"],
        "date_entrainement": _meta["date_entrainement"],
        "modele": _meta["modele"],
        "features_attendues": _meta["colonnes_attendues"],
        "seuil_decision": _meta["seuil_decision"],
        "couts_metier": _meta["couts_metier"],
        "performances_test": _meta["performances_test"],
    }