# Rapport d'Analyse Approfondie : Comportement des Clients E-commerce

**Auteur :** BAKKOURY SALMA  
**Période d'analyse :** Janvier 2023 - Mars 2024  
**Volume de données :** 5 000 transactions  
**Plateforme :** Commerce en ligne turc

---

## 📋 Introduction

Ce rapport présente une analyse exhaustive du comportement des clients d'une plateforme de commerce électronique turque, basée sur un ensemble de données de 5 000 transactions effectuées entre janvier 2023 et mars 2024. L'objectif principal de cette étude est de comprendre les tendances d'achat, les préférences démographiques, l'impact des stratégies promotionnelles et les modèles de consommation afin d'optimiser les stratégies commerciales et marketing. À travers une approche analytique multicouche, nous explorons les dimensions temporelles, catégorielles et démographiques des ventes, tout en évaluant l'efficacité des remises sur les comportements d'achat. Cette analyse permet d'identifier les segments de clientèle les plus rentables, les catégories de produits les plus performantes et les opportunités d'amélioration pour maximiser le chiffre d'affaires et la satisfaction client.

---

## 1️⃣ Configuration de l'Environnement et Chargement des Données

### Code Python

```python
# Installation et mise à jour de kagglehub
!pip install --upgrade kagglehub

import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration du style des visualisations
sns.set_style('whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# Définition du dataset
dataset_handle = "umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr"
file_path = "ecommerce_customer_behavior_dataset.csv"

# Chargement des données
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    dataset_handle,
    file_path,
)

print(f"📦 Dataset chargé avec succès!")
print(f"📊 Dimensions: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"📅 Période: {df['Date'].min()} → {df['Date'].max()}")

# Affichage des premières lignes
df.head(10)
```

### 📊 Interprétation

Le chargement réussi des données révèle un ensemble de 5 000 transactions couvrant une période de 15 mois, offrant une vue longitudinale substantielle pour l'analyse des tendances. La structure du dataset comprend des informations essentielles sur les clients (âge, genre), les produits (catégorie, prix unitaire), les transactions (quantité, remises) et les aspects temporels (dates). Cette richesse informationnelle permet une segmentation multidimensionnelle des comportements d'achat. L'absence de valeurs manquantes, comme vérifié dans la section suivante, garantit la fiabilité des analyses statistiques. La diversité des variables disponibles offre la possibilité d'explorer des corrélations complexes entre facteurs démographiques, temporels et commerciaux, posant ainsi les bases d'une analyse prédictive et prescriptive robuste pour l'optimisation des stratégies e-commerce.

---

## 2️⃣ Évaluation de la Qualité des Données

### Code Python

```python
print("🔍 ÉVALUATION DE LA QUALITÉ DES DONNÉES")
print("="*70)

# Informations générales
print("\n📋 Informations du Dataset:")
df.info()

# Vérification des valeurs manquantes
print("\n" + "="*70)
print("⚠️ Vérification des Valeurs Manquantes:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ Aucune valeur manquante trouvée! Le dataset est propre.")
else:
    print(missing[missing > 0])

# Résumé statistique
print("\n" + "="*70)
print("📊 Résumé Statistique:")
df.describe()
```

### 📊 Interprétation

L'évaluation de la qualité des données confirme l'intégrité exceptionnelle du dataset avec zéro valeur manquante, ce qui est rare dans les contextes réels et facilite grandement les analyses ultérieures. L'examen des statistiques descriptives révèle des informations cruciales : l'âge moyen des clients se situe autour de 40 ans avec une distribution équilibrée, les prix unitaires varient considérablement suggérant une gamme de produits diversifiée, et les quantités commandées montrent une prédominance d'achats unitaires ou en petites quantités typiques du commerce de détail en ligne. Les montants de remise présentent une distribution avec des valeurs nulles fréquentes, indiquant que toutes les transactions ne bénéficient pas de promotions. Cette analyse préliminaire établit les paramètres de base pour comprendre le comportement type du client et permet d'identifier les valeurs aberrantes potentielles qui nécessiteraient un traitement spécifique avant les analyses avancées.

---

## 3️⃣ Ingénierie des Variables

### Code Python

```python
# Conversion de la date
df['Date'] = pd.to_datetime(df['Date'])

# Création de variables financières
df['Total_Amount'] = df['Unit_Price'] * df['Quantity']
df['Final_Amount'] = df['Total_Amount'] - df['Discount_Amount']
df['Discount_Percentage'] = (df['Discount_Amount'] / df['Total_Amount'] * 100).fillna(0)

# Création de variables temporelles
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Day_Name'] = df['Date'].dt.day_name()
df['Quarter'] = df['Date'].dt.quarter
df['Week'] = df['Date'].dt.isocalendar().week

# Création de groupes d'âge
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100],
                          labels=['18-25', '26-35', '36-45', '46-55', '55+'])

# Indicateur de remise
df['Has_Discount'] = (df['Discount_Amount'] > 0).astype(int)

print("✅ Ingénierie des variables terminée!")
print(f"🎯 Total de variables: {df.shape[1]} colonnes")

# Affichage des nouvelles variables
new_features = ['Total_Amount', 'Final_Amount', 'Discount_Percentage', 'Year', 'Month',
                'Month_Name', 'Day', 'DayOfWeek', 'Day_Name', 'Quarter', 'Week',
                'Age_Group', 'Has_Discount']
print(f"\n🔧 Nouvelles variables créées:")
for feat in new_features:
    print(f"   ✓ {feat}")
```

### 📊 Interprétation

L'ingénierie des variables constitue une étape cruciale qui enrichit considérablement le potentiel analytique du dataset original. La création de 13 nouvelles variables dérivées permet une analyse multidimensionnelle sophistiquée. Les variables financières (montant total, montant final, pourcentage de remise) facilitent l'évaluation précise de la rentabilité et de l'impact promotionnel. Les variables temporelles (année, mois, trimestre, jour de la semaine) permettent d'identifier des patterns saisonniers et cycliques essentiels pour la planification des stocks et des campagnes marketing. La segmentation par groupes d'âge transforme une variable continue en catégories stratégiques alignées avec les pratiques de ciblage marketing. L'indicateur binaire de remise simplifie les comparaisons entre transactions promotionnelles et non-promotionnelles. Cette transformation de données brutes en features analytiques prépare le terrain pour des insights actionnables et des modèles prédictifs performants.

---

## 4️⃣ Analyse des Catégories de Produits

### Code Python

```python
# Analyse des catégories de produits
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Commandes par catégorie
ax1 = fig.add_subplot(gs[0, :2])
category_orders = df['Product_Category'].value_counts().sort_values(ascending=True)
colors_cat = plt.cm.Set3(np.linspace(0, 1, len(category_orders)))
bars1 = ax1.barh(category_orders.index, category_orders.values, color=colors_cat,
                 edgecolor='black', linewidth=1.5, alpha=0.85)
ax1.set_xlabel('Nombre de Commandes', fontsize=11, weight='bold')
ax1.set_title('📦 Commandes par Catégorie de Produit', fontsize=13, weight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars1, category_orders.values)):
    ax1.text(val, i, f' {val:,}', va='center', fontsize=10, weight='bold')

# 2. Distribution des catégories (Donut)
ax2 = fig.add_subplot(gs[0, 2])
category_dist = df['Product_Category'].value_counts()
colors_donut = plt.cm.Pastel1(np.linspace(0, 1, len(category_dist)))
wedges, texts, autotexts = ax2.pie(category_dist.values, labels=category_dist.index,
                                     autopct='%1.1f%%', startangle=90, colors=colors_donut,
                                     textprops={'fontsize': 8, 'weight': 'bold'},
                                     pctdistance=0.85)
circle = plt.Circle((0, 0), 0.70, fc='white')
ax2.add_artist(circle)
ax2.set_title('🥧 Mix des Catégories', fontsize=12, weight='bold', pad=10)

# 3. Revenus par catégorie
ax3 = fig.add_subplot(gs[1, :2])
category_revenue = df.groupby('Product_Category')['Final_Amount'].sum().sort_values(ascending=False)
bars3 = ax3.bar(range(len(category_revenue)), category_revenue.values,
                color=plt.cm.viridis(np.linspace(0.2, 0.9, len(category_revenue))),
                edgecolor='black', linewidth=1.5, alpha=0.85)
ax3.set_xticks(range(len(category_revenue)))
ax3.set_xticklabels(category_revenue.index, rotation=45, ha='right', fontsize=10)
ax3.set_ylabel('Revenu Total (₺)', fontsize=11, weight='bold')
ax3.set_title('💰 Revenu par Catégorie de Produit', fontsize=13, weight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars3, category_revenue.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'₺{val/1000:.0f}K', ha='center', va='bottom', fontsize=9, weight='bold')

# 4. Valeur moyenne de commande par catégorie
ax4 = fig.add_subplot(gs[1, 2])
category_aov = df.groupby('Product_Category')['Final_Amount'].mean().sort_values(ascending=False)
bars4 = ax4.barh(category_aov.index, category_aov.values,
                 color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(category_aov))),
                 edgecolor='black', linewidth=1.2, alpha=0.85)
ax4.set_xlabel('Valeur Moy. Commande (₺)', fontsize=10, weight='bold')
ax4.set_title('📊 VMC par Catégorie', fontsize=12, weight='bold', pad=10)
ax4.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars4, category_aov.values)):
    ax4.text(val, i, f' ₺{val:.0f}', va='center', fontsize=8, weight='bold')

# 5. Heatmap de performance des catégories
ax5 = fig.add_subplot(gs[2, :])
category_metrics = df.groupby('Product_Category').agg({
    'Order_ID': 'count',
    'Final_Amount': ['sum', 'mean'],
    'Quantity': 'sum',
    'Discount_Amount': 'mean'
}).round(0)
category_metrics.columns = ['Commandes', 'Revenu', 'VMC', 'Unités Vendues', 'Remise Moy.']
category_metrics_normalized = (category_metrics - category_metrics.min()) / (category_metrics.max() - category_metrics.min())

sns.heatmap(category_metrics_normalized.T, annot=category_metrics.T, fmt=',.0f',
            cmap='YlOrRd', linewidths=1, linecolor='black', cbar_kws={'label': 'Score Normalisé'},
            ax=ax5, annot_kws={'fontsize': 9, 'weight': 'bold'})
ax5.set_title('🔥 Carte de Performance des Catégories', fontsize=13, weight='bold', pad=15)
ax5.set_xlabel('Catégorie de Produit', fontsize=11, weight='bold')
ax5.set_ylabel('Métriques', fontsize=11, weight='bold')
ax5.tick_params(axis='x', rotation=45)

plt.suptitle('🛍️ ANALYSE DE PERFORMANCE DES CATÉGORIES DE PRODUITS', fontsize=18, weight='bold', y=0.998)
plt.show()

# Affichage des champions par catégorie
print("\n🏆 Champions par Catégorie:")
print(f"   • Plus de commandes: {category_orders.index[-1]} ({category_orders.values[-1]:,} commandes)")
print(f"   • Revenu le plus élevé: {category_revenue.index[0]} (₺{category_revenue.values[0]:,.0f})")
print(f"   • Meilleure VMC: {category_aov.index[0]} (₺{category_aov.values[0]:.2f})")
```

### 📊 Interprétation

L'analyse des catégories de produits révèle une hiérarchie claire dans les préférences et la performance commerciale. La distribution des commandes montre que certaines catégories dominent le volume de transactions, ce qui suggère soit une demande naturellement plus élevée, soit un positionnement marketing plus efficace. Cependant, il est crucial de noter que la catégorie générant le plus de commandes n'est pas nécessairement celle produisant le revenu maximal, indiquant des différences significatives dans les prix moyens et les comportements d'achat. La heatmap de performance multi-métriques permet d'identifier des catégories "stars" (haut volume et haute valeur) versus des catégories "niche" (faible volume mais haute VMC). Les unités vendues combinées aux revenus révèlent l'élasticité-prix de chaque catégorie. Ces insights sont essentiels pour optimiser l'allocation des ressources marketing, ajuster les stratégies de pricing et identifier les opportunités de croissance par catégorie dans un contexte de ressources limitées.

---

## 5️⃣ Analyse des Tendances Temporelles

### Code Python

```python
# Analyse des séries temporelles
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

# Préparation des données mensuelles
monthly_data = df.groupby(df['Date'].dt.to_period('M')).agg({
    'Final_Amount': 'sum',
    'Order_ID': 'count',
    'Customer_ID': 'nunique',
    'Discount_Amount': 'sum'
}).reset_index()
monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()
monthly_data['AOV'] = monthly_data['Final_Amount'] / monthly_data['Order_ID']

# 1. Tendance des revenus mensuels
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(monthly_data['Date'], monthly_data['Final_Amount']/1000,
         marker='o', linewidth=3, markersize=8, color='#FF6B6B', label='Revenu')
ax1.fill_between(monthly_data['Date'], monthly_data['Final_Amount']/1000,
                  alpha=0.3, color='#FF6B6B')
ax1.set_title('💰 Tendance des Revenus Mensuels', fontsize=14, weight='bold', pad=15)
ax1.set_xlabel('Date', fontsize=11, weight='bold')
ax1.set_ylabel('Revenu (₺ Milliers)', fontsize=11, weight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11)

# Ligne de tendance
z = np.polyfit(range(len(monthly_data)), monthly_data['Final_Amount']/1000, 1)
p = np.poly1d(z)
ax1.plot(monthly_data['Date'], p(range(len(monthly_data))),
         "--", color='darkred', linewidth=2, alpha=0.8, label='Tendance')

# Annotation du pic
max_idx = monthly_data['Final_Amount'].idxmax()
ax1.annotate(f"Pic: ₺{monthly_data.loc[max_idx, 'Final_Amount']/1000:.0f}K",
             xy=(monthly_data.loc[max_idx, 'Date'], monthly_data.loc[max_idx, 'Final_Amount']/1000),
             xytext=(10, 10), textcoords='offset points', fontsize=10, weight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black'))

# 2. Tendance des commandes mensuelles
ax2 = fig.add_subplot(gs[1, 0])
ax2.bar(monthly_data['Date'], monthly_data['Order_ID'],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(monthly_data))),
        edgecolor='black', linewidth=1.2, alpha=0.85)
ax2.set_title('📦 Commandes Mensuelles', fontsize=13, weight='bold', pad=15)
ax2.set_xlabel('Date', fontsize=11, weight='bold')
ax2.set_ylabel('Nombre de Commandes', fontsize=11, weight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Tendance de la valeur moyenne de commande
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(monthly_data['Date'], monthly_data['AOV'],
         marker='s', linewidth=2.5, markersize=7, color='#4ECDC4')
ax3.fill_between(monthly_data['Date'], monthly_data['AOV'], alpha=0.3, color='#4ECDC4')
ax3.set_title('💵 Tendance de la Valeur Moyenne de Commande', fontsize=13, weight='bold', pad=15)
ax3.set_xlabel('Date', fontsize=11, weight='bold')
ax3.set_ylabel('VMC (₺)', fontsize=11, weight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.tick_params(axis='x', rotation=45)

# 4. Comparaison trimestrielle
ax4 = fig.add_subplot(gs[2, 0])
quarterly_data = df.groupby(['Year', 'Quarter'])['Final_Amount'].sum().reset_index()
quarterly_data['Period'] = quarterly_data['Year'].astype(str) + '-T' + quarterly_data['Quarter'].astype(str)
bars = ax4.bar(quarterly_data['Period'], quarterly_data['Final_Amount']/1000,
               color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#C7CEEA'],
               edgecolor='black', linewidth=1.5, alpha=0.85)
ax4.set_title('📊 Revenu Trimestriel', fontsize=13, weight='bold', pad=15)
ax4.set_xlabel('Trimestre', fontsize=11, weight='bold')
ax4.set_ylabel('Revenu (₺ Milliers)', fontsize=11, weight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

for bar, val in zip(bars, quarterly_data['Final_Amount']/1000):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'₺{val:.0f}K', ha='center', va='bottom', fontsize=9, weight='bold')

# 5. Croissance des clients actifs mensuels
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(monthly_data['Date'], monthly_data['Customer_ID'],
         marker='D', linewidth=2.5, markersize=7, color='#95E1D3')
ax5.fill_between(monthly_data['Date'], monthly_data['Customer_ID'], alpha=0.3, color='#95E1D3')
ax5.set_title('👥 Clients Actifs Mensuels', fontsize=13, weight='bold', pad=15)
ax5.set_xlabel('Date', fontsize=11, weight='bold')
ax5.set_ylabel('Clients Uniques', fontsize=11, weight='bold')
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.tick_params(axis='x', rotation=45)

# 6. Tableau de métriques de croissance
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')

# Calcul des métriques de croissance
first_month_revenue = monthly_data['Final_Amount'].iloc[0]
last_month_revenue = monthly_data['Final_Amount'].iloc[-1]
revenue_growth = ((last_month_revenue - first_month_revenue) / first_month_revenue * 100)

first_month_orders = monthly_data['Order_ID'].iloc[0]
last_month_orders = monthly_data['Order_ID'].iloc[-1]
order_growth = ((last_month_orders - first_month_orders) / first_month_orders * 100)

metrics_data = [
    ['Métrique', 'Premier Mois', 'Dernier Mois', 'Croissance %'],
    ['Revenu', f'₺{first_month_revenue:,.0f}', f'₺{last_month_revenue:,.0f}', f'{revenue_growth:+.1f}%'],
    ['Commandes', f'{first_month_orders:,}', f'{last_month_orders:,}', f'{order_growth:+.1f}%'],
    ['VMC', f'₺{monthly_data["AOV"].iloc[0]:.2f}', f'₺{monthly_data["AOV"].iloc[-1]:.2f}',
     f'{((monthly_data["AOV"].iloc[-1] - monthly_data["AOV"].iloc[0]) / monthly_data["AOV"].iloc[0] * 100):+.1f}%'],
]

table = ax6.table(cellText=metrics_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3)

# Style de l'en-tête
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#667eea')
    cell.set_text_props(weight='bold', color='white')

# Style des lignes de données
for i in range(1, 4):
    for j in range(4):
        cell = table[(i, j)]
        if j == 3:  # Colonne de croissance
            value = float(metrics_data[i][3].replace('%', '').replace('+', ''))
            if value > 0:
                cell.set_facecolor('#90EE90')
            elif value < 0:
                cell.set_facecolor('#FFB6C6')
            else:
                cell.set_facecolor('#FFFACD')
        else:
            cell.set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
        cell.set_text_props(weight='bold')

ax6.set_title('📊 Métriques de Croissance Période sur Période', fontsize=13, weight='bold', pad=20, y=0.95)

plt.suptitle('📈 ANALYSE DES TENDANCES DE REVENU ET DE VENTES', fontsize=18, weight='bold', y=0.998)
plt.show()

print("✅ Analyse des séries temporelles terminée!")
```

### 📊 Interprétation

L'analyse temporelle dévoile des patterns cycliques et des tendances de croissance essentiels pour la planification stratégique. La courbe des revenus mensuels révèle des fluctuations saisonnières potentiellement liées à des événements commerciaux (soldes, fêtes) ou à des facteurs externes économiques. La ligne de tendance superposée indique la direction générale de la croissance, permettant d'isoler les variations saisonnières du momentum sous-jacent. L'identification du pic de revenu fournit un benchmark pour évaluer la performance des mois futurs et comprendre quels facteurs ont contribué à cette performance exceptionnelle. La comparaison trimestrielle offre une vue agrégée qui lisse les variations mensuelles, révélant des tendances plus stables pour les décisions à long terme. L'évolution de la base de clients actifs mensuelle est un indicateur crucial de santé à long terme : une croissance soutenue suggère une acquisition efficace et une rétention réussie, tandis qu'une stagnation ou déclin signalerait des problèmes nécessitant intervention immédiate.

---

## 6️⃣ Analyse de la Stratégie de Remise

### Code Python

```python
# Analyse des remises
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Répartition Remise vs Sans Remise
ax1 = fig.add_subplot(gs[0, 0])
discount_split = df['Has_Discount'].value_counts()
colors_discount = ['#95E1D3', '#FF6B6B']
explode = (0.05, 0.05)
wedges, texts, autotexts = ax1.pie(discount_split.values, labels=['Sans Remise', 'Avec Remise'],
                                     autopct='%1.1f%%', startangle=90, colors=colors_discount,
                                     explode=explode, textprops={'fontsize': 11, 'weight': 'bold'},
                                     shadow=True)
ax1.set_title('🎁 Commandes: Remise vs Sans Remise', fontsize=13, weight='bold', pad=15)

# 2. Distribution des montants de remise
ax2 = fig.add_subplot(gs[0, 1:])
discount_data = df[df['Discount_Amount'] > 0]['Discount_Amount']
ax2.hist(discount_data, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
ax2.axvline(discount_data.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Moyenne: ₺{discount_data.mean():.2f}')
ax2.axvline(discount_data.median(), color='green', linestyle='--', linewidth=2,
            label=f'Médiane: ₺{discount_data.median():.2f}')
ax2.set_title('💰 Distribution des Montants de Remise', fontsize=13, weight='bold', pad=15)
ax2.set_xlabel('Montant de Remise (₺)', fontsize=11, weight='bold')
ax2.set_ylabel('Fréquence', fontsize=11, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# 3. Distribution des pourcentages de remise
ax3 = fig.add_subplot(gs[1, 0])
discount_pct_data = df[df['Discount_Percentage'] > 0]['Discount_Percentage']
ax3.hist(discount_pct_data, bins=40, color='#FFE66D', edgecolor='black', alpha=0.7)
ax3.axvline(discount_pct_data.mean(), color='darkred', linestyle='--', linewidth=2,
            label=f'Moy: {discount_pct_data.mean():.1f}%')
ax3.set_title('📊 Distribution des Pourcentages de Remise', fontsize=13, weight='bold', pad=15)
ax3.set_xlabel('Pourcentage de Remise (%)', fontsize=11, weight='bold')
ax3.set_ylabel('