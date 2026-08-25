# Prédiction du churn client — Telco Customer Churn

Projet final du module *Supervised Learning* (Master IA). De la donnée brute à
une API de scoring, en justifiant chaque décision.

---

## Le problème

Un opérateur télécom veut identifier chaque mois les clients susceptibles de
résilier, afin que le service marketing leur adresse en priorité une offre de
rétention.

**Ce qui rend le problème intéressant n'est pas la classification, mais
l'asymétrie des coûts.** Un client perdu emporte sa valeur résiduelle
(~350 €) ; une offre envoyée à tort ne coûte qu'une remise (~40 €). Un faux
négatif vaut donc environ **neuf faux positifs**, ce qui déplace le seuil de
décision optimal de 0,5 à **0,1026** — et divise le coût par deux.

## Les données

| | |
|---|---|
| Source | Telco Customer Churn (IBM Sample), 7 043 clients × 21 colonnes |
| Cible | `Churn` — le client a-t-il résilié ? (26,54 % de oui) |
| Récupération | `python scripts/download_data.py` (non versionné, MD5 vérifié) |

Deux pièges documentés en Mission 1 : `TotalCharges` contient 11 chaînes vides
correspondant à des clients dont `tenure == 0` — imputées par la **constante 0**
et non par la médiane ; et 22 lignes deviennent identiques sans `customerID`,
mais ce sont des clients distincts au profil commun, **à ne pas supprimer**.

## Le modèle

**Régression logistique** dans un pipeline scikit-learn, avec la feature dérivée
`tenure_bucket`. Ce choix est le résultat d'un benchmark, pas un point de départ :
elle devance forêt aléatoire, gradient boosting, k-NN et Naive Bayes, et égale un
boosting optimisé par 80 essais Optuna — tout en restant interprétable, ce
qu'exigeait le cadrage.

### Performances sur le jeu de test (1 409 clients, jamais utilisé avant)

| Métrique | Valeur |
|---|---|
| **Coût métier** | **19,89 €/client** (baseline naïve : 92,90 € → **−78,6 %**) |
| ROC-AUC | 0,845 |
| AUC-PR | 0,650 |
| Brier | 0,136 |
| Rappel | 0,941 |
| Précision | 0,409 |
| Matrice de confusion | VP 352 · FN 22 · FP 508 · VN 527 |

Le coût en test (19,89 €) est cohérent avec la validation croisée (20,22 €) :
aucun sur-apprentissage. La cible fixée *a priori* en Mission 0 — ≤ 65 €/client —
est largement atteinte.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

Python ≥ 3.10 requis. Versions épinglées dans `requirements.txt`.

## Entraîner le modèle

```bash
python -m src.train
```

Produit `model/pipeline_churn.joblib` et `model/metadata.json`, et vérifie que
les prédictions après rechargement sont **strictement** identiques.

> Lancer depuis la racine du projet. Le pipeline sérialisé référence
> `src.features` : exécuter `python src/train.py` casserait le rechargement.

## Lancer l'API

```bash
uvicorn api.main:app --reload
```

Documentation interactive : <http://127.0.0.1:8000/docs>

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```
```json
{"statut": "ok", "modele_charge": true, "version": "1.0.0"}
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":2,
       "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
       "TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
       "Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":85.5,"TotalCharges":171.0}'
```
```json
{"probabilite_churn": 0.6943, "cibler": true,
 "seuil_applique": 0.1026, "version_modele": "1.0.0"}
```

Un client fidèle (68 mois, contrat deux ans, prélèvement automatique) obtient
`0.0074` et `cibler: false`. Une modalité inconnue renvoie **422**.

### `GET /model-info`

Features attendues, seuil, coûts métier et performances de validation.

## Tests

```bash
pytest -v
```

Huit tests sur le modèle **rechargé depuis le disque**, donc sur l'objet que sert
réellement l'API : forme de la sortie, probabilités dans [0, 1], gestion des
valeurs manquantes, rejet d'une colonne absente, non-régression de la performance,
déterminisme de la sérialisation, cohérence métier (l'engagement doit réduire le
risque), et les trois endpoints.

## Structure

```
├── api/main.py              API FastAPI (3 endpoints)
├── data/raw/                données (non versionnées)
├── model/                   pipeline sérialisé + métadonnées
├── notebooks/
│   ├── 01_eda.ipynb                 M1 — exploration, détection de fuite
│   ├── 02_pipeline_baseline.ipynb   M2 — pipeline, baseline, features
│   ├── 03_benchmark.ipynb           M3 — 5 modèles, Wilcoxon, erreurs
│   └── 04_optimisation_shap.ipynb   M4 — Optuna, calibration, SHAP, seuil
├── reports/
│   ├── 00_cadrage.md        M0 — problème, coûts, métrique, seuil
│   ├── model_card.md        carte du modèle
│   └── monitoring.md        plan de surveillance en production
├── scripts/download_data.py
├── src/{features,train}.py
└── tests/test_pipeline.py
```

## Limites connues

Le modèle **ne détecte pas les churners engagés** : rappel de 0,00 sur les
contrats de deux ans (9 churners manqués sur 336 clients). Plus généralement, les
départs manqués concernent des clients anciens, engagés, en prélèvement
automatique — dont la cause de départ (déménagement, offre concurrente, incident
de service) est **absente des données**. Voir `reports/model_card.md`.