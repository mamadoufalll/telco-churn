"""Régénère les figures du rapport dans reports/figures/.

Usage (depuis la racine du projet) :
    python -m scripts.export_figures

Les figures sont reconstruites depuis les données plutôt que capturées
depuis les notebooks : elles restent ainsi cohérentes avec le code, et
une modification du pipeline se répercute automatiquement.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src.features import (COUT_FN, RANDOM_STATE, SEUIL_THEORIQUE,
                          charger_donnees, construire_pipeline, cout_metier)

warnings.filterwarnings("ignore")

DOSSIER = Path("reports/figures")
DONNEES = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def enregistrer(fig, nom: str) -> None:
    chemin = DOSSIER / f"{nom}.png"
    fig.savefig(chemin, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  {chemin}")


def main() -> None:
    DOSSIER.mkdir(parents=True, exist_ok=True)
    X, y = charger_donnees(DONNEES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    brut = pd.read_csv(DONNEES)

    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    pipe = construire_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        X_train, tenure_bucket=True)
    proba = cross_val_predict(pipe, X_train, y_train, cv=cv10,
                              method="predict_proba")[:, 1]

    print("Génération des figures :")

    # --- 1. Hypothèses de la Mission 1 ------------------------------------
    cible = (brut["Churn"] == "Yes").astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        taux = cible.groupby(brut[col]).mean().sort_values()
        taux.plot.barh(ax=ax, color="indianred")
        ax.axvline(cible.mean(), color="black", ls="--", lw=1)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("taux de churn")
        ax.set_ylabel("")
    fig.suptitle("Taux de churn par modalité (trait : moyenne globale 26,5 %)", y=1.04)
    enregistrer(fig, "01_hypotheses")

    # --- 2. Ancienneté ----------------------------------------------------
    tranches = pd.cut(brut["tenure"], [-1, 6, 12, 24, 48, 72],
                      labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    cible.groupby(tranches, observed=True).mean().plot.bar(
        ax=axes[0], color="indianred", rot=0)
    axes[0].axhline(cible.mean(), color="black", ls="--", lw=1)
    axes[0].set_title("Churn par tranche d'ancienneté (mois)")
    axes[0].set_ylabel("taux de churn")
    for label, groupe in brut.groupby("Churn"):
        axes[1].hist(groupe["tenure"], bins=36, alpha=0.6,
                     label=f"Churn = {label}", density=True)
    axes[1].set_title("Distribution de l'ancienneté selon la cible")
    axes[1].set_xlabel("tenure (mois)")
    axes[1].legend()
    enregistrer(fig, "02_anciennete")

    # --- 3. Benchmark des modèles ----------------------------------------
    scoring = {"cout": make_scorer(cout_metier, response_method="predict_proba",
                                   greater_is_better=False)}
    modeles = {
        "RegLog": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=25),
        "Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "Forêt": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                        random_state=RANDOM_STATE),
        "NaiveBayes": GaussianNB(),
    }
    plis = {}
    for nom, modele in modeles.items():
        r = cross_validate(construire_pipeline(modele, X_train, tenure_bucket=True),
                           X_train, y_train, cv=cv10, scoring=scoring, n_jobs=-1)
        plis[nom] = -r["test_cout"]
    ordre = sorted(plis, key=lambda k: plis[k].mean())

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.boxplot([plis[m] for m in ordre], tick_labels=ordre)
    for i, m in enumerate(ordre, start=1):
        ax.scatter(np.full(len(plis[m]), i), plis[m], alpha=0.5, s=16,
                   color="darkorange")
    ax.set_ylabel("coût métier (€/client)")
    ax.set_title("Benchmark sur 10 plis identiques")
    enregistrer(fig, "03_benchmark")

    # --- 4. Calibration ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibration parfaite")
    frac, moy = calibration_curve(y_train, proba, n_bins=10, strategy="quantile")
    ax.plot(moy, frac, "o-", color="steelblue", label="modèle (probabilités brutes)")
    ax.set_xlabel("probabilité prédite")
    ax.set_ylabel("fréquence observée")
    ax.set_title("Diagramme de fiabilité")
    ax.legend()
    enregistrer(fig, "04_calibration")

    # --- 5. Seuil de décision --------------------------------------------
    seuils = np.linspace(0.01, 0.90, 300)
    couts = np.array([cout_metier(y_train, proba, s) for s in seuils])
    i_opt = int(np.argmin(couts))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(seuils, couts, color="steelblue")
    ax.axvline(seuils[i_opt], color="green", ls="--",
               label=f"optimum empirique {seuils[i_opt]:.4f}")
    ax.axvline(SEUIL_THEORIQUE, color="red", ls=":",
               label=f"théorique {SEUIL_THEORIQUE:.4f}")
    ax.axvline(0.5, color="grey", ls="-.", label="défaut 0,5")
    ax.set_xlabel("seuil de décision")
    ax.set_ylabel("coût métier (€/client)")
    ax.set_title("Le seuil pèse plus que le modèle")
    ax.legend()
    enregistrer(fig, "05_seuil")

    # --- 6. Analyse d'erreurs --------------------------------------------
    pred = (proba >= SEUIL_THEORIQUE).astype(int)
    typ = np.select(
        [(y_train.values == 1) & (pred == 1), (y_train.values == 1) & (pred == 0),
         (y_train.values == 0) & (pred == 1), (y_train.values == 0) & (pred == 0)],
        ["VP", "FN", "FP", "VN"], default="?")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for t, couleur in [("VP", "indianred"), ("FN", "darkred"),
                       ("FP", "orange"), ("VN", "steelblue")]:
        axes[0].hist(proba[typ == t], bins=30, alpha=0.5, label=t,
                     density=True, color=couleur)
    axes[0].axvline(SEUIL_THEORIQUE, color="black", ls="--")
    axes[0].set_xlabel("probabilité prédite")
    axes[0].set_title("Probabilités par type de prédiction")
    axes[0].legend()
    axes[1].boxplot([X_train["tenure"].values[typ == t] for t in ["VP", "FN", "FP", "VN"]],
                    tick_labels=["VP", "FN", "FP", "VN"])
    axes[1].set_ylabel("ancienneté (mois)")
    axes[1].set_title("Les churners manqués sont des clients anciens")
    enregistrer(fig, "06_erreurs")

    # --- 7. Importance SHAP ----------------------------------------------
    import shap
    pipe.fit(X_train, y_train)
    Z = pipe.named_steps["pretraitement"].transform(
        pipe.named_steps["enrichissement"].transform(X_train))
    noms = [n.split("__", 1)[1]
            for n in pipe.named_steps["pretraitement"].get_feature_names_out()]
    valeurs = shap.LinearExplainer(pipe.named_steps["modele"], Z,
                                   feature_names=noms)(Z)
    importance = pd.Series(np.abs(valeurs.values).mean(0), index=noms) \
                   .sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.head(10).sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("|valeur SHAP| moyenne")
    ax.set_title("Importance globale des variables")
    enregistrer(fig, "07_shap")

    print(f"\n7 figures enregistrées dans {DOSSIER}/")


if __name__ == "__main__":
    main()