# 📊 Rapport d’Analyse Exploratoire du Dataset Wine Quality

## Introduction

L’objectif de cette analyse est d’explorer le dataset *Wine Quality* de l’UCI Machine Learning Repository, composé de données physico-chimiques de vins blancs et d’un score de qualité attribué par des experts. À travers plusieurs visualisations (distribution de la qualité, histogrammes et matrice de corrélation), nous cherchons à comprendre les relations entre les variables et à identifier les facteurs influençant le plus la qualité du vin.

Cette analyse exploratoire constitue une étape essentielle avant la construction d’un modèle prédictif, car elle permet d’identifier les tendances, les valeurs extrêmes, les variables pertinentes et les dépendances entre les dimensions du dataset.

---

## 1. Distribution de la qualité du vin

Le graphique de distribution montre que :

- La majorité des vins ont une qualité comprise entre **5 et 7**, avec un pic à **6**.
- Les vins de très haute qualité (8–9) ou très basse qualité (3–4) sont rares.
- La distribution est **déséquilibrée**, ce qui pourrait influencer les futurs modèles prédictifs.

### ✔ Commentaire  
Cette concentration autour de valeurs moyennes indique que le dataset contient peu d’exemples extrêmes. Cela limite les analyses fines sur les vins exceptionnels et nécessite une gestion du déséquilibre lors de la modélisation (ex. : repondération ou techniques de sur-échantillonnage).

---

## 2. Analyse des distributions des variables physico-chimiques

Quatre variables ont été analysées via histogrammes :  
- **Alcohol**  
- **Volatile acidity**  
- **Citric acid**  
- **Residual sugar**

### 🔹 Alcohol  
Distribution asymétrique, principalement entre 9 % et 12 %.  
**Commentaire :** Associé positivement à la qualité. Les vins plus alcoolisés sont souvent mieux notés.

### 🔹 Volatile Acidity  
Concentrée à de faibles niveaux, avec quelques valeurs extrêmes.  
**Commentaire :** Une acidité volatile élevée est un facteur qui dégrade fortement la qualité (goût vinaigré).

### 🔹 Citric Acid  
Distribution centrée autour de 0.2 – 0.4 g/dm³.  
**Commentaire :** Améliore la fraîcheur et contribue à la qualité du vin.

### 🔹 Residual Sugar  
Très forte asymétrie avec présence de valeurs extrêmement élevées.  
**Commentaire :** Ce paramètre ne corrèle pas fortement avec la qualité mais reflète différentes typologies de vins.

---

## 3. Analyse de la matrice de corrélation

La heatmap met en évidence les relations entre les variables et la qualité du vin.

### 🔸 Corrélations positives avec la qualité :
- **Alcohol (~ +0.44)** → meilleure variable prédictive.  
- Légères corrélations avec **sulphates** et **citric acid**.

### 🔸 Corrélations négatives :
- **Density (~ –0.31)** → vins moins denses = meilleure qualité.  
- **Chlorides (~ –0.20)**.  
- **Volatile acidity (~ –0.19)** → très significative.

### ✔ Commentaire  
Ces corrélations montrent que :
- Un vin léger, faiblement acide et avec un taux d’alcool plus élevé est généralement mieux noté.  
- Certaines variables (pH, sucre résiduel) ont un impact assez faible, ce qui permet de concentrer les modèles sur les variables les plus explicatives.

---

## Conclusion

Cette analyse exploratoire du dataset *Wine Quality* a permis de dégager plusieurs enseignements clés :

1. La qualité du vin est principalement centrée autour de valeurs moyennes (5 à 7).
2. Les distributions des variables physico-chimiques montrent des asymétries et la présence de valeurs extrêmes.
3. Les variables **alcohol**, **density** et **volatile acidity** sont les plus fortement corrélées avec la qualité.
4. Certaines caractéristiques ont un impact limité, ce qui simplifie le choix des variables pour la modélisation.
5. Le dataset est déséquilibré, ce qui devra être pris en compte pour développer un modèle prédictif fiable.

Ces résultats constituent une base solide pour poursuivre un travail de modélisation ou approfondir l’étude des facteurs influençant la qualité du vin.

---

