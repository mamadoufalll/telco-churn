"""Entraîne le modèle final, l'évalue sur le jeu de test et le sérialise.

Usage (depuis la racine du projet) :
    python -m src.train

Produit deux fichiers dans model/ :
    - pipeline_churn.joblib : le pipeline complet (enrichissement → prétraitement → modèle)
    - metadata.json         : version, seuil, features attendues, performances

Le jeu de test n'est évalué qu'ici, une seule fois, à l'issue de toutes les
décisions de modélisation. Toute autre utilisation en ferait un second jeu de
validation et invaliderait l'estimation de performance.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.features import (COUT_FN, COUT_FP, RANDOM_STATE, SEUIL_THEORIQUE,
                      charger_donnees, construire_pipeline, cout_metier)

VERSION = "1.0.0"
CHEMIN_DONNEES = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DOSSIER_MODELE = Path("model")
FEATURES = dict(tenure_bucket=True)   # retenu en M2, confirmé en M3


def main() -> None:
    X, y = charger_donnees(str(CHEMIN_DONNEES))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipeline = construire_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        X_train, **FEATURES)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    prediction = (proba >= SEUIL_THEORIQUE).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_test, prediction).ravel()

    performances = {
        "cout_metier_eur_par_client": round(cout_metier(y_test, proba), 2),
        "cout_baseline_naive": round(float(y_test.mean()) * COUT_FN, 2),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "auc_pr": round(average_precision_score(y_test, proba), 4),
        "brier": round(brier_score_loss(y_test, proba), 4),
        "rappel": round(recall_score(y_test, prediction), 4),
        "precision": round(precision_score(y_test, prediction), 4),
        "f1": round(f1_score(y_test, prediction), 4),
        "matrice_confusion": {"VP": int(vp), "FN": int(fn), "FP": int(fp), "VN": int(vn)},
        "part_parc_ciblee": round(float(prediction.mean()), 4),
    }

    metadonnees = {
        "version": VERSION,
        "date_entrainement": date.today().isoformat(),
        "modele": "LogisticRegression (max_iter=2000)",
        "features_derivees": FEATURES,
        "seuil_decision": round(SEUIL_THEORIQUE, 4),
        "couts_metier": {"faux_negatif_eur": COUT_FN, "faux_positif_eur": COUT_FP},
        "colonnes_attendues": list(X.columns),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "taux_churn_train": round(float(y_train.mean()), 4),
        "performances_test": performances,
    }

    DOSSIER_MODELE.mkdir(exist_ok=True)
    joblib.dump(pipeline, DOSSIER_MODELE / "pipeline_churn.joblib")
    (DOSSIER_MODELE / "metadata.json").write_text(
        json.dumps(metadonnees, indent=2, ensure_ascii=False), encoding="utf-8")

    # Vérification exigée par l'énoncé : les prédictions après rechargement
    # doivent être strictement identiques, pas seulement proches.
    recharge = joblib.load(DOSSIER_MODELE / "pipeline_churn.joblib")
    identiques = np.array_equal(proba, recharge.predict_proba(X_test)[:, 1])

    print(f"Modèle v{VERSION} entraîné sur {len(X_train):,} clients.")
    print(f"Coût sur le test : {performances['cout_metier_eur_par_client']} €/client "
          f"(baseline naïve : {performances['cout_baseline_naive']} €)")
    print(f"ROC-AUC {performances['roc_auc']} | rappel {performances['rappel']} "
          f"| précision {performances['precision']}")
    print(f"Prédictions identiques après rechargement : {identiques}")
    if not identiques:
        raise SystemExit("Sérialisation non déterministe — modèle non publiable.")


if __name__ == "__main__":
    main()