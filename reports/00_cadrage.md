# Mission 0 — Cadrage du problème

> Rédigé **avant toute modélisation**. La métrique principale et les seuils de
> réussite fixés ici engagent l'ensemble du projet et ne seront pas révisés a
> posteriori en fonction des résultats obtenus.

## 1. Problème métier

**Objectif business.** Identifier chaque mois les clients présentant un risque
élevé de résiliation, afin de leur adresser en priorité une offre de rétention
avant leur départ.

**Qui exploite la prédiction.** Le service marketing. La sortie du modèle n'est
pas une fin en soi : elle produit une **liste de ciblage** pour une campagne de
rétention (contact + remise commerciale). C'est le coût de cette campagne,
rapporté au chiffre d'affaires préservé, qui arbitre tous les choix techniques
qui suivent.

**Définition de la cible.** `Churn` = le client a résilié son abonnement
(oui / non). Le dataset IBM fournit un **instantané** : l'état de résiliation au
moment de l'extraction, sans fenêtre temporelle explicite. On ne distingue donc
pas « a churné dans les 30 jours » de « a churné un jour ». C'est une limite
structurelle du jeu de données, à garder en tête pour l'interprétation
opérationnelle et pour la question du concept drift.

## 2. Coûts d'erreur

Ancrage tarifaire (statistiques descriptives seules, aucune information sur la
cible) : abonnement moyen **64,76 €/mois**, ancienneté médiane **29 mois**, et
**55 % des clients en contrat mensuel** — donc résiliables sans friction
contractuelle.

| Erreur | Situation | Chiffrage | Coût retenu |
|---|---|---|---|
| **Faux négatif (FN)** | Un client sur le point de partir n'est pas ciblé : il résilie. | Marge brute résiduelle perdue : 64,76 € × 30 % de marge × 18 mois d'horizon | **350 €** |
| **Faux positif (FP)** | Une offre de rétention part vers un client qui serait resté. | Remise consentie inutilement : 64,76 € × 20 % × 3 mois, + ~2 € de coût de contact | **40 €** |

**Lequel est le plus grave ?** Le faux négatif, d'un facteur **~9**. Un client
perdu emporte l'intégralité de sa valeur résiduelle et devra être remplacé par
une acquisition coûteuse ; un faux positif ne coûte qu'une remise ponctuelle,
partiellement récupérée par l'effet de fidélisation qu'elle produit sur un
client déjà satisfait.

**Hypothèses assumées.** Horizon de valeur fixé à 18 mois (choix médian :
12 mois sous-estimerait la valeur d'un parc à ancienneté médiane de 29 mois,
24 mois supposerait une visibilité irréaliste sur un marché concurrentiel).
Taux de marge de 30 %, ordre de grandeur usuel du revenu récurrent télécom.
Ces deux paramètres sont explicites précisément pour pouvoir être discutés :
faire varier l'horizon de 12 à 24 mois déplace le ratio de coûts de 6:1 à 12:1,
sans changer la conclusion qualitative.

## 3. Métrique

**Métrique principale — coût métier moyen par client, à minimiser :**

```
Coût moyen = (350 € × FN + 40 € × FP) / n
```

C'est la seule métrique qui encode explicitement le ratio de coûts établi
ci-dessus. Elle traduit directement l'objectif business, et fournit à la
Mission 4 le critère d'optimisation du seuil de décision.

**Métriques secondaires surveillées :**

- **Rappel** sur la classe *churn* — quelle proportion des partants réels est
  effectivement captée. C'est la grandeur que le métier comprend immédiatement.
- **AUC-PR** — invariante au seuil et plus informative que l'AUC-ROC lorsque la
  classe positive est minoritaire (~26,5 %), car elle ne bénéficie pas du grand
  nombre de vrais négatifs.

**Métriques écartées, et pourquoi.** L'**accuracy** est trompeuse : la baseline
naïve « personne ne churne » atteint déjà 73,5 %. Le **F1** pondère précision et
rappel à égalité, ce qui contredirait frontalement le ratio de coûts de 9:1.

**Seuils de réussite fixés a priori.**

| Critère | Cible | Référence |
|---|---|---|
| Coût métier moyen | **≤ 65 €/client** | Baseline naïve : 0,2654 × 350 € = **92,89 €/client**, soit une réduction exigée de 30 % |
| Rappel (churn) | **≥ 0,75** | Trois partants sur quatre doivent être atteints par la campagne |
| AUC-ROC | **≥ 0,82** | Repère de performance usuel sur ce jeu de données |

**Prédiction sur le seuil de décision.** Le seuil optimal théorique pour une
matrice de coûts asymétrique vaut :

```
seuil* = C_FP / (C_FP + C_FN) = 40 / (40 + 350) ≈ 0,10
```

Soit très en deçà du 0,5 par défaut — cohérent avec le fait qu'il vaut mieux
cibler large que manquer un partant. La Mission 4 vérifiera empiriquement cette
valeur sur les données de validation.

## 4. Risques et hypothèses

**Stationnarité — hypothèse fragile.** Rien ne garantit que la distribution
d'aujourd'hui soit celle de demain : une offre agressive d'un concurrent, une
révision de la grille tarifaire ou l'effet même de la campagne de rétention
modifient le comportement de churn. Le modèle est donc supposé valable à court
terme et devra être réentraîné périodiquement — d'où le dispositif de
monitoring prévu en Mission 5.

**Latence — contrainte faible.** Le scoring s'effectue en batch mensuel sur le
parc client. Aucune exigence de temps réel : le coût computationnel n'est pas un
critère de sélection du modèle.

**RGPD — vigilance requise.** La base légale invoquée est l'intérêt légitime
(prévention de la perte de clientèle). Le traitement suppose la minimisation des
données et une information des personnes concernées. Point sensible : le jeu de
données contient `gender` et `SeniorCitizen`, deux attributs protégés. Leur
usage doit être interrogé — un ciblage marketing différencié selon l'âge ou le
genre soulève un risque de discrimination indirecte, examiné dans la question de
réflexion sur l'équité.

**Explicabilité — exigence opérationnelle.** Le service marketing doit pouvoir
justifier pourquoi un client donné figure sur la liste, tant pour calibrer
l'argumentaire commercial que pour répondre à une éventuelle demande
d'explication. Cette contrainte oriente vers des modèles interprétables ou
accompagnés d'explications post-hoc : c'est l'objet de l'analyse SHAP en
Mission 4.