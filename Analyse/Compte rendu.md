<img src="SETTAT.png" style="height:100px;margin-right:95px"/>  

# Rapport d'Analyse Approfondie du Comportement Client E-commerce

**Auteur :** BAKKOURY SALMA  

<img src="Photo salma.jpg" style="height:200px;margin-right:150px"/>

**Date :** Novembre 2025  
**Source des données :** Dataset Kaggle - E-commerce Customer Behavior and Sales Analysis (Turquie)

## Introduction

Le commerce électronique connaît une croissance exponentielle à l'échelle mondiale, transformant radicalement les habitudes d'achat des consommateurs. Dans ce contexte hautement compétitif, la compréhension approfondie du comportement client devient un avantage stratégique crucial pour optimiser les performances commerciales et améliorer l'expérience utilisateur.

Ce rapport présente une analyse complète d'un ensemble de données contenant 5 000 transactions e-commerce provenant d'une plateforme de vente en ligne turque, couvrant la période de janvier 2023 à mars 2024. L'objectif principal est d'extraire des insights actionnables concernant les préférences produits, les tendances d'achat, l'efficacité des stratégies de remise, et les caractéristiques démographiques des clients. Cette analyse permettra d'identifier les leviers d'optimisation des revenus et d'améliorer la stratégie commerciale globale.

---

## 1. Configuration de l'Environnement et Chargement des Données

### 1.1 Installation et Importation des Bibliothèques

```python
# Installation de kagglehub pour accéder aux données
!pip install --upgrade kagglehub

# Importation des bibliothèques nécessaires
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

print("✅ Bibliothèques chargées avec succès!")
print("🎨 Visualisations améliorées activées!")
```

### 1.2 Chargement du Dataset

```python
# Définition du chemin du dataset
dataset_handle = "umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr"
file_path = "ecommerce_customer_behavior_dataset.csv"

# Chargement des données
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    dataset_handle,
    file_path,
)

print(f"📦 Dataset chargé avec succès!")
print(f"📊 Dimensions : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"📅 Période : {df['Date'].min()} → {df['Date'].max()}")
print(f"\n{'='*70}\n")

# Aperçu des données
df.head(10)
```

**Interprétation :** Le dataset a été chargé avec succès depuis Kaggle et contient 5 000 transactions e-commerce. Cette taille d'échantillon est suffisamment robuste pour identifier des tendances significatives tout en restant gérable pour une analyse approfondie. La période couverte de 15 mois permet d'observer les évolutions temporelles et les saisonnalités potentielles dans le comportement d'achat.

---

## 2. Évaluation de la Qualité des Données

```python
print("🔍 ÉVALUATION DE LA QUALITÉ DES DONNÉES")
print("="*70)

# Informations de base
print("\n📋 Informations sur le Dataset :")
df.info()

print("\n" + "="*70)
print("⚠ Vérification des Valeurs Manquantes :")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ Aucune valeur manquante détectée ! Dataset propre.")
else:
    print(missing[missing > 0])

print("\n" + "="*70)
print("📊 Résumé Statistique :")
df.describe()
```

**Interprétation :** L'évaluation initiale de la qualité des données révèle un dataset remarquablement propre, sans valeurs manquantes. Cette intégrité des données est essentielle pour garantir la fiabilité des analyses ultérieures. Les statistiques descriptives permettent d'identifier les plages de valeurs pour chaque variable numérique, ce qui aide à détecter d'éventuelles anomalies ou valeurs aberrantes. La présence de données complètes pour toutes les transactions témoigne d'une collecte de données rigoureuse et d'une gestion de base de données efficace.

---

## 3. Ingénierie des Caractéristiques

```python
# Conversion de la date en format datetime
df['Date'] = pd.to_datetime(df['Date'])

# Création de caractéristiques financières
df['Total_Amount'] = df['Unit_Price'] * df['Quantity']
df['Final_Amount'] = df['Total_Amount'] - df['Discount_Amount']
df['Discount_Percentage'] = (df['Discount_Amount'] / df['Total_Amount'] * 100).fillna(0)

# Extraction de caractéristiques temporelles
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

print("✅ Ingénierie des caractéristiques terminée !")
print(f"🎯 Total des caractéristiques : {df.shape[1]} colonnes")
print(f"\n🔧 Nouvelles caractéristiques créées :")
new_features = ['Total_Amount', 'Final_Amount', 'Discount_Percentage', 'Year', 'Month',
                'Month_Name', 'Day', 'DayOfWeek', 'Day_Name', 'Quarter', 'Week',
                'Age_Group', 'Has_Discount']
for feat in new_features:
    print(f"   ✓ {feat}")
```

**Interprétation :** L'ingénierie des caractéristiques enrichit considérablement le dataset en créant 13 nouvelles variables dérivées. Les caractéristiques financières (montant total, montant final, pourcentage de remise) permettent d'analyser la rentabilité et l'impact des promotions. Les variables temporelles extraites (année, mois, jour de la semaine, trimestre) sont cruciales pour identifier les patterns saisonniers et les tendances cycliques. La segmentation par groupe d'âge facilite l'analyse démographique et le ciblage marketing. Ces transformations préparent les données pour des analyses multidimensionnelles plus sophistiquées.

---

## 4. Analyse des Catégories de Produits

```python
# Analyse complète des catégories de produits
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Commandes par catégorie
ax1 = fig.add_subplot(gs[0, :2])
category_orders = df['Product_Category'].value_counts().sort_values(ascending=True)
colors_cat = plt.cm.Set3(np.linspace(0, 1, len(category_orders)))
bars1 = ax1.barh(category_orders.index, category_orders.values, color=colors_cat,
                 edgecolor='black', linewidth=1.5, alpha=0.85)
ax1.set_xlabel('Nombre de Commandes', fontsize=11, weight='bold')
ax1.set_title('📦 Commandes par Catégorie de Produits', fontsize=13, weight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars1, category_orders.values)):
    ax1.text(val, i, f' {val:,}', va='center', fontsize=10, weight='bold')

# 2. Distribution des catégories (Donut Chart)
ax2 = fig.add_subplot(gs[0, 2])
category_dist = df['Product_Category'].value_counts()
colors_donut = plt.cm.Pastel1(np.linspace(0, 1, len(category_dist)))
wedges, texts, autotexts = ax2.pie(category_dist.values, labels=category_dist.index,
                                     autopct='%1.1f%%', startangle=90, colors=colors_donut,
                                     textprops={'fontsize': 8, 'weight': 'bold'},
                                     pctdistance=0.85)
circle = plt.Circle((0, 0), 0.70, fc='white')
ax2.add_artist(circle)
ax2.set_title('🥧 Mix de Catégories', fontsize=12, weight='bold', pad=10)

# 3. Revenus par catégorie
ax3 = fig.add_subplot(gs[1, :2])
category_revenue = df.groupby('Product_Category')['Final_Amount'].sum().sort_values(ascending=False)
bars3 = ax3.bar(range(len(category_revenue)), category_revenue.values,
                color=plt.cm.viridis(np.linspace(0.2, 0.9, len(category_revenue))),
                edgecolor='black', linewidth=1.5, alpha=0.85)
ax3.set_xticks(range(len(category_revenue)))
ax3.set_xticklabels(category_revenue.index, rotation=45, ha='right', fontsize=10)
ax3.set_ylabel('Revenus Totaux (₺)', fontsize=11, weight='bold')
ax3.set_title('💰 Revenus par Catégorie de Produits', fontsize=13, weight='bold', pad=15)
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

# 5. Heatmap de performance par catégorie
ax5 = fig.add_subplot(gs[2, :])
category_metrics = df.groupby('Product_Category').agg({
    'Order_ID': 'count',
    'Final_Amount': ['sum', 'mean'],
    'Quantity': 'sum',
    'Discount_Amount': 'mean'
}).round(0)
category_metrics.columns = ['Commandes', 'Revenus', 'VMC', 'Unités Vendues', 'Remise Moy.']
category_metrics_normalized = (category_metrics - category_metrics.min()) / (category_metrics.max() - category_metrics.min())

sns.heatmap(category_metrics_normalized.T, annot=category_metrics.T, fmt=',.0f',
            cmap='YlOrRd', linewidths=1, linecolor='black', cbar_kws={'label': 'Score Normalisé'},
            ax=ax5, annot_kws={'fontsize': 9, 'weight': 'bold'})
ax5.set_title('🔥 Heatmap de Performance des Catégories', fontsize=13, weight='bold', pad=15)
ax5.set_xlabel('Catégorie de Produits', fontsize=11, weight='bold')
ax5.set_ylabel('Métriques', fontsize=11, weight='bold')
ax5.tick_params(axis='x', rotation=45)

plt.suptitle('🛍️ ANALYSE DE PERFORMANCE DES CATÉGORIES DE PRODUITS', 
             fontsize=18, weight='bold', y=0.998)
plt.show()

# Résumé des champions par catégorie
print("\n🏆 Champions par Catégorie :")
print(f"   • Plus de commandes : {category_orders.index[-1]} ({category_orders.values[-1]:,} commandes)")
print(f"   • Revenus les plus élevés : {category_revenue.index[0]} (₺{category_revenue.values[0]:,.0f})")
print(f"   • Meilleure VMC : {category_aov.index[0]} (₺{category_aov.values[0]:.2f})")
```

**Interprétation :** L'analyse des catégories de produits révèle une distribution relativement équilibrée des commandes entre les différentes catégories, suggérant un catalogue diversifié et attrayant pour différents segments de clientèle. La heatmap de performance permet d'identifier les catégories leaders et celles nécessitant une attention particulière. Les disparités entre le volume de commandes et les revenus générés indiquent que certaines catégories, bien que moins populaires en termes de volume, génèrent une valeur moyenne de commande plus élevée, ce qui pourrait refléter des produits premium ou à marge supérieure. Cette analyse multidimensionnelle aide à prioriser les investissements marketing et l'optimisation des stocks.

---

## 5. Analyse des Tendances Temporelles et des Revenus

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
         marker='o', linewidth=3, markersize=8, color='#FF6B6B', label='Revenus')
ax1.fill_between(monthly_data['Date'], monthly_data['Final_Amount']/1000,
                  alpha=0.3, color='#FF6B6B')
ax1.set_title('💰 Tendance des Revenus Mensuels', fontsize=14, weight='bold', pad=15)
ax1.set_xlabel('Date', fontsize=11, weight='bold')
ax1.set_ylabel('Revenus (₺ Milliers)', fontsize=11, weight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11)

# Ajout de la ligne de tendance
z = np.polyfit(range(len(monthly_data)), monthly_data['Final_Amount']/1000, 1)
p = np.poly1d(z)
ax1.plot(monthly_data['Date'], p(range(len(monthly_data))),
         "--", color='darkred', linewidth=2, alpha=0.8, label='Tendance')

# Annotation du pic et du creux
max_idx = monthly_data['Final_Amount'].idxmax()
min_idx = monthly_data['Final_Amount'].idxmin()
ax1.annotate(f"Pic : ₺{monthly_data.loc[max_idx, 'Final_Amount']/1000:.0f}K",
             xy=(monthly_data.loc[max_idx, 'Date'], monthly_data.loc[max_idx, 'Final_Amount']/1000),
             xytext=(10, 10), textcoords='offset points', fontsize=10, weight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black'))

# 2. Commandes mensuelles
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
ax4.set_title('📊 Revenus Trimestriels', fontsize=13, weight='bold', pad=15)
ax4.set_xlabel('Trimestre', fontsize=11, weight='bold')
ax4.set_ylabel('Revenus (₺ Milliers)', fontsize=11, weight='bold')
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

# 6. Tableau des métriques de croissance
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
    ['Revenus', f'₺{first_month_revenue:,.0f}', f'₺{last_month_revenue:,.0f}', f'{revenue_growth:+.1f}%'],
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
        if j == 3:  # Colonne croissance
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

ax6.set_title('📊 Métriques de Croissance Période-à-Période', 
              fontsize=13, weight='bold', pad=20, y=0.95)

plt.suptitle('📈 ANALYSE DES TENDANCES DES REVENUS ET VENTES', 
             fontsize=18, weight='bold', y=0.998)
plt.show()

print("✅ Analyse des séries temporelles terminée !")
```

**Interprétation :** L'analyse temporelle des revenus révèle des patterns cycliques et des tendances de croissance importantes. Les fluctuations mensuelles peuvent être attribuées à des facteurs saisonniers, des campagnes promotionnelles ou des événements spéciaux. La ligne de tendance générale indique la direction globale de la performance commerciale sur la période étudiée. La comparaison trimestrielle permet d'identifier les périodes de haute et basse saison, informations cruciales pour la planification des stocks et des campagnes marketing. L'évolution de la valeur moyenne de commande et du nombre de clients actifs mensuels fournit des indicateurs clés sur la santé financière et la fidélisation client. Le tableau de croissance période-à-période quantifie précisément l'évolution des métriques principales, facilitant l'évaluation de la performance et la fixation d'objectifs futurs.

---

## 6. Analyse de la Stratégie de Remise

```python
# Analyse des remises
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Remise vs Sans Remise
ax1 = fig.add_subplot(gs[0, 0])
discount_split = df['Has_Discount'].value_counts()
colors_discount = ['#95E1D3', '#FF6B6B']
explode = (0.05, 0.05)
wedges, texts, autotexts = ax1.pie(discount_split.values, labels=['Sans Remise', 'Avec Remise'],
                                     autopct='%1.1f%%', startangle=90, colors=colors_discount,
                                     explode=explode, textprops={'fontsize': 11, 'weight': 'bold'},
                                     shadow=True)
ax1.set_title('🎁 Commandes : Avec vs Sans Remise', fontsize=13, weight='bold', pad=15)

# 2. Distribution du montant de remise
ax2 = fig.add_subplot(gs[0, 1:])
discount_data = df[df['Discount_Amount'] > 0]['Discount_Amount']
ax2.hist(discount_data, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
ax2.axvline(discount_data.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Moyenne : ₺{discount_data.mean():.2f}')
ax2.axvline(discount_data.median(), color='green', linestyle='--', linewidth=2,
            label=f'Médiane : ₺{discount_data.median():.2f}')
ax2.set_title('💰 Distribution des Montants de Remise', fontsize=13, weight='bold', pad=15)
ax2.set_xlabel('Montant de Remise (₺)', fontsize=11, weight='bold')
ax2.set_ylabel('Fréquence', fontsize=11, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# 3. Distribution du pourcentage de remise
ax3 = fig.add_subplot(gs[1, 0])
discount_pct_data = df[df['Discount_Percentage'] > 0]['Discount_Percentage']
ax3.hist(discount_pct_data, bins=40, color='#FFE66D', edgecolor='black', alpha=0.7)
ax3.axvline(discount_pct_data.mean(), color='darkred', linestyle='--', linewidth=2,
            label=f'Moy. : {discount_pct_data.mean():.1f}%')
ax3.set_title('📊 Distribution des Pourcentages de Remise', fontsize=13, weight='bold', pad=15)
ax3.set_xlabel('Pourcentage de Remise (%)', fontsize=11, weight='bold')
ax3.set_ylabel('Fréquence', fontsize=11, weight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# 4. Remise par catégorie
ax4 = fig.add_subplot(gs[1, 1])
category_discount = df.groupby('Product_Category').agg({
    'Discount_Amount': 'mean',
    'Has_Discount': 'mean'
}).sort_values('Discount_Amount', ascending=False)

bars = ax4.barh(category_discount.index, category_discount['Discount_Amount'],
                color=plt.cm.plasma(np.linspace(0.2, 0.9, len(category_discount))),
                edgecolor='black', linewidth=1.2, alpha=0.85)
ax4.set_xlabel('Remise Moyenne (₺)', fontsize=11, weight='bold')
ax4.set_title('🏷️ Remise Moyenne par Catégorie', fontsize=13, weight='bold', pad=15)
ax4.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, category_discount['Discount_Amount'])):
    ax4.text(val, i, f' ₺{val:.1f}', va='center', fontsize=9, weight='bold')

# 5. Analyse d'impact
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

# Calcul des métriques
orders_with_discount = df[df['Has_Discount'] == 1]
orders_without_discount = df[df['Has_Discount'] == 0]

avg_order_with_discount = orders_with_discount['Final_Amount'].mean()
avg_order_without_discount = orders_without_discount['Final_Amount'].mean()

total_revenue = df['Final_Amount'].sum()
total_discounts = df['Discount_Amount'].sum()
discount_rate = (total_discounts / total_revenue * 100)

impact_text = f"""
╔═══════════════════════════╗
║  RAPPORT D'IMPACT REMISES  ║
╠═══════════════════════════╣
║
║ 📊 Métriques Globales :
║   • Total Remises : ₺{total_discounts:,.0f}
║   • Taux de Remise : {discount_rate:.2f}%
║   • Commandes avec remise : {len(orders_with_discount):,}
║   • Commandes sans remise : {len(orders_without_discount):,}
║
║ 💰 Impact sur la Valeur de Commande :
║   • VMC (avec remise) : ₺{avg_order_with_discount:.2f}
║   • VMC (sans remise) : ₺{avg_order_without_discount:.2f}
║   • Différence : ₺{avg_order_with_discount - avg_order_without_discount:.2f}
║
║ 🎯 Insights Clés :
║   • Remise Moyenne : ₺{df['Discount_Amount'].mean():.2f}
║   • Remise Maximale : ₺{df['Discount_Amount'].max():.2f}
║   • % Remise : {discount_pct_data.mean():.1f}% moy.
║
╚═══════════════════════════╝
"""

ax5.text(0.1, 0.5, impact_text, fontsize=10, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))

plt.suptitle('🎁 ANALYSE DE LA STRATÉGIE DE REMISE', fontsize=18, weight='bold', y=0.995)
plt.show()

print("✅ Analyse des remises terminée !")
```

**Interprétation :** L'analyse de la stratégie de remise démontre l'impact significatif des promotions sur le comportement d'achat. La proportion de commandes bénéficiant de remises indique l'intensité de la stratégie promotionnelle. La distribution des montants et pourcentages de remise révèle les tactiques de pricing adoptées, avec une concentration autour de certains seuils psychologiques. La comparaison de la valeur moyenne de commande entre les transactions avec et sans remise permet d'évaluer l'efficacité des promotions pour stimuler les ventes. Si la VMC avec remise est supérieure, cela suggère que les remises encouragent l'achat de produits plus chers ou en quantité supérieure, validant ainsi la stratégie. À l'inverse, une VMC inférieure pourrait indiquer une érosion de marge nécessitant un réajustement. L'analyse par catégorie identifie les segments où les remises sont les plus généreuses, informant sur les stratégies de compétitivité par produit.

---

## 7. Analyses Avancées et Corrélations

```python
# Analyses avancées
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. Matrice de corrélation
ax1 = fig.add_subplot(gs[0, :])
numerical_cols = ['Age', 'Unit_Price', 'Quantity', 'Discount_Amount',
                  'Total_Amount', 'Final_Amount', 'Discount_Percentage']
correlation = df[numerical_cols].corr()

mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=2, linecolor='black',
            cbar_kws={"shrink": 0.8, "label": "Coefficient de Corrélation"},
            ax=ax1, annot_kws={'fontsize': 10, 'weight': 'bold'})
ax1.set_title('🔗 Matrice de Corrélation des Caractéristiques', fontsize=14, weight='bold', pad=15)
ax1.tick_params(axis='x', rotation=45)
ax1.tick_params(axis='y', rotation=0)

# 2. Prix unitaire vs Quantité
ax2 = fig.add_subplot(gs[1, 0])
scatter1 = ax2.scatter(df['Unit_Price'], df['Quantity'],
                       c=df['Final_Amount'], cmap='viridis',
                       s=50, alpha=0.5, edgecolors='black', linewidth=0.5)
ax2.set_title('💰 Prix Unitaire vs Quantité', fontsize=13, weight='bold', pad=15)
ax2.set_xlabel('Prix Unitaire (₺)', fontsize=11, weight='bold')
ax2.set_ylabel('Quantité', fontsize=11, weight='bold')
ax2.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax2)
cbar1.set_label('Montant Final (₺)', fontsize=10, weight='bold')

# 3. Âge vs Dépenses
ax3 = fig.add_subplot(gs[1, 1])
age_spending = df.groupby('Age')['Final_Amount'].mean().reset_index()
ax3.scatter(age_spending['Age'], age_spending['Final_Amount'],
            c=age_spending['Final_Amount'], cmap='plasma',
            s=100, alpha=0.7, edgecolors='black', linewidth=1)
z = np.polyfit(age_spending['Age'], age_spending['Final_Amount'], 2)
p = np.poly1d(z)
ax3.plot(age_spending['Age'], p(age_spending['Age']),
         "--", color='red', linewidth=2.5, alpha=0.8, label='Tendance')
ax3.set_title('👤 Âge vs Dépenses Moyennes', fontsize=13, weight='bold', pad=15)
ax3.set_xlabel('Âge', fontsize=11, weight='bold')
ax3.set_ylabel('Dépenses Moyennes (₺)', fontsize=11, weight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.suptitle('🔬 ANALYSES AVANCÉES & CORRÉLATIONS', fontsize=18, weight='bold', y=0.995)
plt.show()

print("\n📊 Corrélations Clés :")
print("="*50)
# Identification des corrélations les plus fortes
corr_pairs = []
for i in range(len(correlation.columns)):
    for j in range(i+1, len(correlation.columns)):
        corr_pairs.append({
            'Caractéristique 1': correlation.columns[i],
            'Caractéristique 2': correlation.columns[j],
            'Corrélation': correlation.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs).sort_values('Corrélation', key=abs, ascending=False)
print("\nTop 5 des Corrélations les Plus Fortes :")
for idx, row in corr_df.head(5).iterrows():
    print(f"   • {row['Caractéristique 1']} ↔ {row['Caractéristique 2']}: {row['Corrélation']:.3f}")
```

**Interprétation :** Les analyses de corrélation avancées révèlent les relations entre les différentes variables du dataset, permettant d'identifier les facteurs prédictifs et les dynamiques sous-jacentes du comportement d'achat. Les corrélations fortes, qu'elles soient positives ou négatives, indiquent des associations significatives entre variables. Par exemple, une corrélation élevée entre le montant total et le montant final est attendue et naturelle, tandis qu'une corrélation entre l'âge et les dépenses révélerait des patterns démographiques intéressants pour la segmentation client. Le nuage de points Prix-Quantité illustre les stratégies d'achat, montrant si les clients achètent en volume lorsque les prix sont bas. L'analyse Âge-Dépenses identifie les segments démographiques les plus rentables, guidant les efforts de ciblage marketing. Ces insights multivariés sont essentiels pour développer des modèles prédictifs et des stratégies personnalisées.

---

## Conclusion


En conclusion, cette analyse fournit une base factuelle robuste pour la prise de décision stratégique. Les recommandations issues de ces insights incluent : l'optimisation du mix produit en fonction de la rentabilité par catégorie, l'ajustement des campagnes promotionnelles selon les périodes identifiées comme plus réceptives, et le développement de stratégies de personnalisation basées sur les segments démographiques et comportementaux identifiés. La continuation de telles analyses avec des données actualisées et l'intégration de techniques d'apprentissage automatique permettraient d'affiner encore davantage la compréhension du comportement client et d'anticiper les tendances futures du marché e-commerce.
