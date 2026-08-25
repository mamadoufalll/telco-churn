"""Feature engineering et construction du pipeline de prétraitement.

Ce code vit dans un module (et non dans le notebook) pour une raison précise :
un pipeline sérialisé avec joblib ne stocke pas le code de ses transformateurs,
seulement une *référence* vers eux. Un `FunctionTransformer` qui pointerait vers
une fonction définie dans un notebook serait donc impossible à recharger depuis
l'API de la Mission 5. Toute fonction utilisée dans le pipeline doit être
importable depuis un module stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Coûts métier fixés en Mission 0
COUT_FN = 350.0
COUT_FP = 40.0
SEUIL_THEORIQUE = COUT_FP / (COUT_FP + COUT_FN)  # ≈ 0,1026

SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Écartées des features : identifiant sans pouvoir de généralisation,
# et attribut protégé dont l'information mutuelle mesurée en M1 est nulle.
COLONNES_EXCLUES = ["customerID", "gender"]


def charger_donnees(chemin: str) -> tuple[pd.DataFrame, pd.Series]:
    """Charge le CSV brut et sépare features et cible.

    Seule opération appliquée ici : la conversion de `TotalCharges` en numérique.
    Les 11 chaînes vides deviennent des NaN, imputés *dans le pipeline*.
    """
    df = pd.read_csv(chemin)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"] + COLONNES_EXCLUES)
    return X, y


def enrichir(
    X: pd.DataFrame,
    tenure_bucket: bool = False,
    nb_services: bool = False,
    price_drift: bool = False,
) -> pd.DataFrame:
    """Ajoute les features dérivées activées.

    Aucune statistique n'est apprise ici : chaque feature est une fonction
    déterministe de la ligne courante. La transformation est donc sans risque
    de fuite, même appliquée avant le split.
    """
    X = X.copy()

    if tenure_bucket:
        # H2 : la relation ancienneté/churn est décroissante mais non linéaire.
        X["tenure_bucket"] = pd.cut(
            X["tenure"], [-1, 6, 12, 24, 48, 72],
            labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
        ).astype(str)

    if nb_services:
        # H5 : six colonnes encodent une information redondante et ordonnable.
        X["nb_services"] = (X[SERVICES] == "Yes").sum(axis=1)

    if price_drift:
        # Rapport entre facture actuelle et facture moyenne historique.
        # > 1 signale une hausse tarifaire récente — moteur de mécontentement.
        moyenne_historique = X["TotalCharges"] / X["tenure"].replace(0, np.nan)
        X["price_drift"] = (
            (X["MonthlyCharges"] / moyenne_historique)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)  # nouveaux clients : pas d'historique, pas de dérive
        )

    return X


def construire_preprocesseur(X: pd.DataFrame) -> ColumnTransformer:
    """Assemble le ColumnTransformer à partir des types de colonnes présents.

    Toutes les statistiques apprises (médiane, moyenne, écart-type, modalités)
    le sont exclusivement lors du `fit`, donc sur le train seul.
    """
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]

    # TotalCharges reçoit son propre traitement : les valeurs absentes
    # correspondent à des clients dont tenure == 0, dont le total facturé
    # vaut 0. Imputer par la médiane leur attribuerait le cumul d'un client
    # ancien — erreur de sens, pas erreur de calcul.
    total_charges = [c for c in num if c == "TotalCharges"]
    autres_num = [c for c in num if c != "TotalCharges"]

    return ColumnTransformer([
        ("total_charges", Pipeline([
            ("imputation", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("echelle", StandardScaler()),
        ]), total_charges),
        ("numerique", Pipeline([
            ("imputation", SimpleImputer(strategy="median")),
            ("echelle", StandardScaler()),
        ]), autres_num),
        ("categoriel", Pipeline([
            ("imputation", SimpleImputer(strategy="most_frequent")),
            ("encodage", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])


def construire_pipeline(modele, X_reference: pd.DataFrame, **features) -> Pipeline:
    """Pipeline complet : enrichissement → prétraitement → modèle.

    `X_reference` sert uniquement à déterminer les types de colonnes après
    enrichissement ; aucune valeur n'en est extraite.
    """
    enrichissement = FunctionTransformer(enrichir, kw_args=features)
    preprocesseur = construire_preprocesseur(enrichir(X_reference, **features))
    return Pipeline([
        ("enrichissement", enrichissement),
        ("pretraitement", preprocesseur),
        ("modele", modele),
    ])


def cout_metier(y_true, proba, seuil: float = SEUIL_THEORIQUE) -> float:
    """Coût métier moyen par client, en euros. À MINIMISER.

    Métrique principale définie en Mission 0 : elle est la seule à encoder
    le ratio de coûts de 9:1 entre faux négatif et faux positif.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= seuil).astype(int)
    faux_negatifs = int(((y_true == 1) & (pred == 0)).sum())
    faux_positifs = int(((y_true == 0) & (pred == 1)).sum())
    return (COUT_FN * faux_negatifs + COUT_FP * faux_positifs) / len(y_true)