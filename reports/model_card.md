# Model card — Prédiction du churn client

**Version 1.0.0** · Entraînée le 25 août 2026 · Régression logistique

---

## 1. Usage prévu

**Ce à quoi le modèle sert.** Produire chaque mois une liste de clients à cibler
par une campagne de rétention, à partir des données contractuelles et de
souscription disponibles dans le système de facturation.

**Qui l'utilise.** Le service marketing, pour prioriser ses actions commerciales.

**Ce à quoi il ne sert pas.** Le modèle ne doit pas être utilisé pour :

- refuser un service, modifier un tarif ou dégrader une offre à un client ;
- prendre une décision produisant un effet juridique ou affectant
  significativement une personne, ce qui relèverait de l'article 22 du RGPD ;
- prédire un comportement individuel avec certitude — la sortie est une
  probabilité, pas un verdict.

**Décision automatisée ?** Non. Le modèle propose une liste ; un opérateur humain
décide de l'offre commerciale envoyée. C'est cette intervention qui maintient le
traitement hors du champ de la décision entièrement automatisée.

## 2. Données d'entraînement

| | |
|---|---|
| Source | Telco Customer Churn (IBM Sample) |
| Volume | 7 043 clients, dont 5 634 en entraînement et 1 409 en test |
| Période | Non documentée par le fournisseur — **limite majeure** |
| Cible | Résiliation, taux de 26,54 % |
| Variables | 18 utilisées après exclusion de `customerID` et `gender` |

**Représentativité.** Le jeu de données est un échantillon fictif fourni par IBM
à des fins pédagogiques. Il ne provient pas d'un opérateur réel identifié, et sa
représentativité d'un parc client européen n'est pas établie. Les montants sont
en dollars et ont été traités comme des euros pour le calcul des coûts — ce qui
n'affecte pas le ratio C_FN / C_FP, seul déterminant du seuil.

**Absence d'horodatage.** La cible est un instantané : « le client a résilié »,
sans fenêtre temporelle. On ne distingue pas un départ à 30 jours d'un départ à
deux ans. Le modèle prédit donc un **état**, pas un délai — ce qui limite sa
valeur pour planifier le moment d'une intervention.

## 3. Performances

Mesurées sur 1 409 clients de test n'ayant servi à **aucune** décision de
modélisation, au seuil de 0,1026.

| Métrique | Valeur | Référence |
|---|---|---|
| **Coût métier** | **19,89 €/client** | Baseline naïve : 92,90 € (**−78,6 %**) |
| ROC-AUC | 0,845 | |
| AUC-PR | 0,650 | Taux de base : 0,265 |
| Brier | 0,136 | |
| Rappel | 0,941 | 352 churners captés sur 374 |
| Précision | 0,409 | |
| Part du parc ciblée | 61,0 % | |

Coût en validation croisée : 20,22 €/client. L'écart avec le test (19,89 €) est
inférieur à la variance inter-plis : **aucun sur-apprentissage détecté**.

**Calibration.** Le modèle est bien calibré sans correction (écart moyen à la
diagonale : 0,008). Ni Platt ni l'isotonique n'améliorent le score de Brier. La
régression logistique optimisant une règle de score propre, ses probabilités sont
directement exploitables pour un classement par risque.

## 4. Performances par sous-groupe

| Sous-groupe | n | Taux de churn réel | Part ciblée | Rappel | Précision | ROC-AUC |
|---|---|---|---|---|---|---|
| Femmes | 687 | 0,281 | 60,7 % | 0,938 | 0,434 | 0,839 |
| Hommes | 722 | 0,251 | 61,4 % | 0,945 | 0,386 | 0,852 |
| Non-senior | 1 187 | 0,233 | 56,7 % | 0,924 | 0,379 | 0,847 |
| **Senior** | **222** | **0,441** | **84,2 %** | **0,990** | 0,519 | **0,777** |
| Contrat mensuel | 773 | 0,426 | 93,0 % | 0,991 | 0,453 | 0,750 |
| Contrat 1 an | 300 | 0,120 | 42,3 % | 0,722 | 0,205 | 0,745 |
| **Contrat 2 ans** | **336** | **0,027** | **4,2 %** | **0,000** | 0,000 | 0,740 |

### Deux constats qui demandent une décision explicite

**Le genre ne crée aucune disparité.** Les taux de ciblage (60,7 % contre 61,4 %)
et les rappels sont quasi identiques. C'était attendu : `gender` a été exclu des
features dès la Mission 1, son information mutuelle avec la cible étant nulle.
L'exclusion servait ici la performance autant que l'équité — cas favorable, et
rare.

**L'âge crée une disparité forte.** Les clients seniors sont ciblés à **84,2 %**
contre 56,7 % pour les autres, soit un ratio de 1,49. La **parité démographique
n'est donc pas respectée**. Trois éléments à verser au débat :

1. La disparité reflète une différence réelle : les seniors churnent à 44,1 %
   contre 23,3 %. Le modèle ne fabrique pas l'écart, il le reproduit.
2. Sous le critère d'**égalité des chances**, le modèle est favorable aux
   seniors : rappel de 0,990 contre 0,924, précision de 0,519 contre 0,379. Un
   senior sur le départ a *plus* de chances d'être retenu.
3. La conséquence du ciblage est de **recevoir une offre commerciale
   avantageuse**, non de subir un refus. Une sur-représentation dans une
   population bénéficiaire n'a pas la même portée éthique qu'une
   sur-représentation dans une population pénalisée.

**Position retenue** : la disparité est acceptable dans cet usage précis, parce
que le ciblage est bénéfique et que l'égalité des chances est respectée. Elle
deviendrait inacceptable si le modèle était réutilisé pour une décision
défavorable — d'où la restriction d'usage en section 1. La performance
discriminante est par ailleurs plus faible sur ce groupe (ROC-AUC 0,777 contre
0,847), ce qui justifie un suivi séparé.

**Angle mort sur les contrats longs.** Le rappel est de **0,000** sur les
contrats de deux ans : les 9 churners de ce segment sont tous manqués. Le modèle
a appris que l'engagement protège — ce qui est vrai à 97,3 % — et n'alerte
jamais sur ce segment. Ces départs sont rares mais concernent des clients à forte
valeur. Une surveillance métier distincte est recommandée, le modèle étant
structurellement aveugle à ce cas.

## 5. Limites

**Les churners manqués partagent un profil précis** : ancienneté médiane de
52 mois contre 9 pour les churners détectés, 77 % en contrat annuel, 69 % en
prélèvement automatique. Ils présentent tous les signaux de fidélité et partent
quand même. **La cause de leur départ n'est pas dans les données** — déménagement,
offre concurrente, incident de service, insatisfaction accumulée. Aucun
raffinement algorithmique ne les rendra détectables : seules de nouvelles
variables le permettraient (tickets support, incidents réseau, évolution de la
consommation, ancienneté du terminal).

**La précision est faible par construction** (0,409) : six clients ciblés sur dix
ne seraient pas partis. Ce n'est pas un défaut mais la conséquence assumée du
ratio de coûts — éviter un départ justifie neuf offres inutiles. Si le budget
marketing ne le permet pas, cibler par rang décroissant de probabilité (voir
Mission 4, tableau de capacité : les 20 % les plus à risque donnent une précision
de 0,68).

**Le modèle est linéaire** : il suppose un effet additif dans l'espace des
log-cotes et ne capte les interactions que si elles sont explicitées. Le
benchmark a montré que cette limite ne coûte rien ici — mais elle pourrait coûter
sur d'autres données.

**Le seuil dépend d'hypothèses de coûts contestables.** C_FN = 350 € suppose un
horizon de valeur de 18 mois et une marge de 30 %. Faire varier l'horizon de 12 à
24 mois déplace le ratio de 6:1 à 12:1, et donc le seuil de 0,077 à 0,143. La
courbe de coût étant très plate dans cette zone, l'impact reste faible — mais
l'hypothèse doit être validée par le contrôle de gestion avant tout déploiement.

## 6. Considérations éthiques et réglementaires

**Base légale** : intérêt légitime (prévention de la perte de clientèle). Le
traitement suppose l'information des personnes concernées et la minimisation des
données.

**Attributs protégés** : `gender` est exclu des features. `SeniorCitizen` est
conservé, car il porte une information prédictive réelle et le ciblage est
bénéfique — mais ce choix est explicite et révisable, et il fait l'objet d'un
suivi séparé (section 4).

**Droit à l'explication** : chaque décision est décomposable en contributions par
variable via SHAP. Le modèle étant linéaire, ces contributions sont exactes et
non approchées, et s'expriment en rapports de cotes compréhensibles par un
conseiller commercial.

**Risque de boucle de rétroaction** : les clients ciblés reçoivent une offre qui
modifie leur comportement. Le prochain jeu d'entraînement contiendra donc des
clients dont le non-départ résulte de l'intervention du modèle lui-même. Sans
précaution, le modèle apprendra que ce profil ne churne pas et cessera de
l'alerter. **Recommandation : conserver un groupe témoin non ciblé** de 5 à 10 %,
seul moyen de mesurer l'effet réel de la campagne et de maintenir un signal
d'entraînement non contaminé.

## 7. Maintenance

Réentraînement trimestriel recommandé, ou immédiat si le monitoring déclenche une
alerte. Voir `reports/monitoring.md`.

**Contact** : responsable du projet, via le dépôt Git.