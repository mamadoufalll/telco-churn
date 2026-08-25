# Plan de monitoring en production

Modèle de churn v1.0.0 · Scoring mensuel en batch

---

## Le problème que le monitoring doit résoudre

La cible n'est observable qu'avec retard. Un client scoré en janvier ne révèle
son départ qu'au bout de plusieurs semaines, voire jamais s'il reste. **Attendre
la vérité terrain pour détecter une dégradation, c'est la détecter trois mois
trop tard.**

Le dispositif s'organise donc en deux temps : des signaux disponibles
immédiatement, et une mesure de performance qui arrive plus tard mais tranche.

## 1. Surveillance immédiate — sans attendre la vérité terrain

### Dérive des données en entrée (*data drift*)

Comparer la distribution des variables scorées ce mois-ci à celle du jeu
d'entraînement.

| Type de variable | Test | Seuil d'alerte |
|---|---|---|
| Numériques (`tenure`, `MonthlyCharges`, `TotalCharges`) | Kolmogorov-Smirnov | p < 0,01 |
| Catégorielles (`Contract`, `InternetService`, `PaymentMethod`…) | Khi-deux d'adéquation | p < 0,01 |
| Toutes | Indice de stabilité de population (PSI) | PSI > 0,2 |

Le PSI est le plus lisible pour un suivi mensuel : en dessous de 0,1 la
population est stable, entre 0,1 et 0,2 elle mérite attention, au-delà de 0,2
elle a changé.

**Priorité aux variables qui pèsent.** Une dérive sur `tenure`, `Contract` ou
`MonthlyCharges` — le trio de tête de l'analyse SHAP — est bien plus grave
qu'une dérive sur `PhoneService`. Les alertes sont pondérées par l'importance
SHAP.

### Dérive des prédictions (*prediction drift*)

Plus rapide à interpréter que la dérive des entrées, car elle agrège tout.

- **Part du parc ciblée** : référence 61,0 %. Alerte si l'écart dépasse
  ±10 points sur un mois, ou ±5 points deux mois de suite.
- **Probabilité moyenne prédite** : référence 0,265, qui doit rester proche du
  taux de churn attendu. Un décrochage signale que la population scorée ne
  ressemble plus à celle d'entraînement.
- **Forme de la distribution** : le modèle produit une distribution étalée. Un
  resserrement autour de la moyenne indiquerait une perte de pouvoir
  discriminant.

### Santé technique

Taux d'erreurs HTTP 5xx, latence au 95e centile, taux de rejets 422 (schéma
d'entrée). Une hausse des 422 signale généralement une modification du système
amont — le signal le plus précoce d'une rupture de contrat de données.

## 2. Surveillance différée — quand la vérité terrain arrive

À chaque fenêtre d'observation close (trimestre), recalculer sur les clients
scorés :

| Indicateur | Référence | Seuil de réentraînement |
|---|---|---|
| **Coût métier moyen** | 19,89 €/client | > 30 € |
| ROC-AUC | 0,845 | < 0,78 |
| AUC-PR | 0,650 | < 0,55 |
| Rappel | 0,941 | < 0,85 |
| Brier | 0,136 | > 0,17 |

**Le coût métier est l'indicateur qui décide.** Les autres servent au diagnostic :
un ROC-AUC stable avec un coût qui monte signale un problème de **calibration ou
de seuil**, pas de pouvoir discriminant — la correction est alors un réajustement
du seuil, bien moins coûteux qu'un réentraînement.

### Calibration dans le temps

Recalculer le diagramme de fiabilité chaque trimestre. Une dérive de calibration
est particulièrement grave ici, puisque le ciblage sous contrainte de budget
repose sur le **classement** par probabilité.

## 3. Le groupe témoin — non négociable

Sans groupe témoin, le monitoring mesure un artefact. Les clients ciblés reçoivent
une offre qui modifie leur comportement : leur taux de départ observé n'est plus
celui qu'ils auraient eu sans intervention. Le modèle paraîtra donc se dégrader
alors qu'il fonctionne — puisque ses vrais positifs cessent de partir.

**Dispositif** : 5 à 10 % des clients au-dessus du seuil sont volontairement
**non ciblés**, tirés au hasard. Ce groupe fournit deux mesures irremplaçables :

1. La **performance réelle du modèle**, non contaminée par l'effet de la campagne.
2. L'**effet causal de l'offre de rétention** — la différence de taux de départ
   entre ciblés et témoins. C'est la seule façon de savoir si la campagne sert à
   quelque chose, question distincte de celle de la qualité du modèle.

Le coût de ce dispositif (quelques départs non prévenus) est très inférieur à
celui de piloter à l'aveugle.

## 4. Politique de réentraînement

| Déclencheur | Action |
|---|---|
| Routine | Réentraînement trimestriel sur les 24 derniers mois glissants |
| PSI > 0,2 sur une variable à forte importance SHAP | Investigation sous 5 jours ouvrés |
| Part ciblée hors de ±10 points | Investigation immédiate |
| Coût métier > 30 €/client | Réentraînement immédiat |
| ROC-AUC stable mais coût dégradé | Réajustement du seuil, sans réentraînement |
| Changement de grille tarifaire ou d'offre | Réentraînement anticipé, sans attendre le trimestre |

**Toute version réentraînée est comparée à la version en production sur les mêmes
plis**, avec un test de Wilcoxon apparié, avant remplacement. Une nouvelle
version n'est déployée que si elle fait au moins aussi bien : la nouveauté n'est
pas un argument.

**Révision des hypothèses de coût.** Le seuil découle de C_FN = 350 € et
C_FP = 40 €. Ces montants doivent être revus annuellement avec le contrôle de
gestion : une évolution de la marge ou du coût d'acquisition déplace le seuil
optimal, indépendamment de toute dérive des données.

## 5. Traçabilité

Journaliser pour chaque prédiction : horodatage, version du modèle, features
reçues, probabilité produite, décision de ciblage, et appartenance ou non au
groupe témoin. Sans ce journal, aucune analyse rétrospective n'est possible et
l'obligation d'explicabilité ne peut être honorée.

Conservation alignée sur la politique de rétention des données clients, avec
pseudonymisation des identifiants.