# 📊 Rapport d'Analyse Approfondie du Comportement Client E-commerce

**Auteur :** BAKKOURY SALMA  
**Date :** Novembre 2025  
**Source des données :** Dataset Kaggle - E-commerce Customer Behavior and Sales Analysis (Turquie)

---

## 📝 Introduction
Dans un contexte où le commerce électronique connaît une croissance exponentielle, la compréhension approfondie du comportement des clients devient un enjeu stratégique majeur pour les entreprises. Ce rapport présente une analyse détaillée de 5 000 transactions réalisées sur une plateforme e-commerce turque entre janvier 2023 et mars 2024.  
L’objectif est d’extraire des insights concernant :  
- les tendances d’achat,  
- la performance des catégories produits,  
- la saisonnalité,  
- l’impact des remises,  
- et le comportement démographique des clients.

Toutes les analyses ci-dessous incluent les **codes Python** issus du notebook Google Colab ainsi que **les interprétations correspondantes**.

---

## 🔧 1. Configuration de l’Environnement et Chargement des Données
```python
!pip install --upgrade kagglehub
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

**Interprétation :**  
On installe et importe les bibliothèques essentielles à l’analyse de données et aux visualisations.

---

## 🔍 2. Exploration et Qualité des Données
```python
df.info()
df.describe()
df.isnull().sum()
```

**Interprétation :**  
Les données sont propres : aucune valeur manquante n’est détectée, ce qui permet une analyse directe.

---

## ⚙️ 3. Ingénierie des Caractéristiques
```python
df['Date'] = pd.to_datetime(df['Date'])
df['Total_Amount'] = df['Unit_Price'] * df['Quantity']
df['Final_Amount'] = df['Total_Amount'] - df['Discount_Amount']
df['Discount_Percentage'] = (df['Discount_Amount'] / df['Total_Amount'] * 100).fillna(0)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Age_Group'] = pd.cut(df['Age'], bins=[0,25,35,45,55,100],
                         labels=['18-25','26-35','36-45','46-55','55+'])
```

**Interprétation :**  
Ces nouvelles variables enrichissent fortement la capacité analytique (finances, temps, segmentation clients).

---

## 🛍️ 4. Analyse des Catégories Produits
```python
category_orders = df['Product_Category'].value_counts()
category_revenue = df.groupby('Product_Category')['Final_Amount'].sum()
category_aov = df.groupby('Product_Category')['Final_Amount'].mean()
```

**Interprétation :**  
Certaines catégories dominent en volume, alors que d’autres génèrent davantage de revenu ou un panier moyen supérieur.

---

## 📈 5. Analyse Temporelle
```python
monthly_data = df.groupby(df['Date'].dt.to_period('M')).agg({
    'Final_Amount':'sum', 'Order_ID':'count'
}).reset_index()
monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()
```

**Interprétation :**  
La série temporelle montre des pics mensuels liés probablement à des promotions ou périodes saisonnières.

---

## 💸 6. Analyse des Remises
```python
discount_rate = df['Discount_Amount'].sum() / df['Final_Amount'].sum() * 100
orders_with_discount = df[df['Has_Discount']==1]
orders_without_discount = df[df['Has_Discount']==0]
```

**Interprétation :**  
Les commandes avec remises représentent une part significative ; elles augmentent le volume mais diminuent le panier moyen.

---

## 👥 7. Analyse Démographique
```python
gender_spending = df.groupby('Gender')['Final_Amount'].agg(['sum','mean'])
age_category = pd.crosstab(df['Age_Group'], df['Product_Category'])
```

**Interprétation :**  
Les préférences diffèrent par genre et par âge ; certaines tranches d'âge dépensent davantage.

---

## 🔬 8. Corrélations
```python
corr = df[['Unit_Price','Quantity','Final_Amount','Discount_Amount']].corr()
```

**Interprétation :**  
Une corrélation forte apparaît entre `Final_Amount` et `Total_Amount`, ce qui est logique. Les remises ont un effet négatif sur la valeur finale.

---

## 🧾 Conclusion
Cette analyse met en évidence plusieurs enseignements clés :
- Les catégories ne contribuent pas toutes de la même manière : certaines apportent du volume, d’autres du revenu.
- Une saisonnalité claire se manifeste dans les ventes mensuelles.
- Les remises stimulent les achats mais réduisent la marge moyenne.
- Les comportements varient selon le genre et les groupes d'âge, ouvrant la voie à un ciblage marketing plus intelligent.
- Les corrélations financières confirment la structure économique du modèle transactionnel.

**Recommandations :**
- Optimiser les remises selon les catégories à forte élasticité.  
- Mener des campagnes ciblées par âge et genre.  
- Renforcer les stocks et promotions durant les mois de pic.  
- Développer un modèle de prédiction du panier moyen ou du churn client.

Ce rapport constitue une base approfondie pour orienter des décisions marketing, financières et opérationnelles.

---

**Fin du rapport.**
