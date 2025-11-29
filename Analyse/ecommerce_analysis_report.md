# 📊 Rapport d'Analyse Approfondie du Comportement Client E-commerce

**Auteur :** BAKKOURY SALMA  
**Date :** Novembre 2025  
**Source des données :** Dataset Kaggle - E-commerce Customer Behavior and Sales Analysis (Turquie)

---

## 📝 Introduction

Dans un contexte où le commerce électronique connaît une croissance exponentielle, la compréhension approfondie du comportement des clients devient un enjeu stratégique majeur pour les entreprises. Ce rapport présente une analyse détaillée de 5 000 transactions réalisées sur une plateforme e-commerce turque entre janvier 2023 et mars 2024. L'objectif principal est d'identifier les tendances d'achat, les préférences des consommateurs, l'efficacité des stratégies promotionnelles et les opportunités d'optimisation commerciale.

L'analyse s'articule autour de plusieurs axes fondamentaux : l'exploration et la qualité des données, la performance des catégories de produits, les tendances temporelles des ventes, l'impact des remises sur le comportement d'achat, et les caractéristiques démographiques des clients. Cette approche multidimensionnelle permet de dégager des insights actionnables pour améliorer la stratégie commerciale et maximiser la rentabilité de la plateforme.

Les résultats présentés dans ce rapport reposent sur des techniques avancées de data science, incluant l'ingénierie des caractéristiques, l'analyse statistique descriptive, la visualisation de données et l'exploration de corrélations. Chaque section combine code Python et interprétations détaillées pour faciliter la compréhension et la réplication de l'analyse.

---

## 🔧 1. Configuration de l'Environnement et Chargement des Données

### 1.1 Installation et Import des Bibliothèques

```python
# Installation de kagglehub pour accéder aux datasets Kaggle
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

print("✅ Bibliothèques chargées avec succès!")
print("🎨 Visualisations améliorées activées!")
```

### 1.2 Chargement du Dataset depuis Kaggle

```python
# Définition du dataset Kaggle
dataset_handle = "umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr"
file_path = "ecommerce_customer_behavior_dataset.csv"

# Chargement des données via kagglehub
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    dataset_handle,
    file_path,
)

print(f"📦 Dataset chargé avec succès!")
print(f"📊 Dimensions: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"📅 Période couverte: {df['Date'].min()} → {df['Date'].max()}")
print("\n" + "="*70 + "\n")

# Affichage des premières lignes
print("Aperçu des données:")
print(df.head(10))
```

**Interprétation :** Cette première étape consiste à établir l'environnement de travail en important toutes les bibliothèques nécessaires pour l'analyse de données et la visualisation. L'utilisation de `kagglehub` permet d'accéder directement aux datasets hébergés sur Kaggle, facilitant ainsi la reproductibilité de l'analyse. Le dataset chargé contient 5 000 transactions, ce qui constitue un échantillon suffisamment robuste pour identifier des tendances significatives et tirer des conclusions fiables. La configuration de Matplotlib et Seaborn avec des styles personnalisés garantit que toutes les visualisations seront esthétiques, claires et professionnelles, facilitant ainsi la communication des résultats.

---

## 🔍 2. Exploration et Évaluation de la Qualité des Données

### 2.1 Analyse de la Structure du Dataset

```python
print("🔍 ÉVALUATION DE LA QUALITÉ DES DONNÉES")
print("="*70)

# Informations détaillées sur le dataset
print("\n📋 Informations sur le Dataset:")
df.info()

print("\n" + "="*70)
print("⚠️ Vérification des Valeurs Manquantes:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ Aucune valeur manquante détectée! Dataset propre et complet.")
else:
    print("Valeurs manquantes par colonne:")
    print(missing[missing > 0])

print("\n" + "="*70)
print("📊 Résumé Statistique des Variables Numériques:")
print(df.describe())

print("\n" + "="*70)
print("📈 Aperçu des Variables Catégorielles:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}: {df[col].nunique()} valeurs uniques")
    print(df[col].value_counts().head())
```

**Interprétation :** L'analyse exploratoire révèle un dataset remarquablement propre et bien structuré, sans aucune valeur manquante, ce qui est exceptionnel dans le domaine de l'analyse de données réelles. Cette qualité élevée des données permet de procéder directement à l'analyse sans nécessiter de phase de nettoyage complexe, économisant ainsi un temps précieux et réduisant les risques d'introduction de biais lors du traitement des valeurs manquantes.

L'examen de la structure du dataset montre une diversité de types de variables : des identifiants (Order_ID, Customer_ID), des variables démographiques (Age, Gender), des informations produits (Product_Category, Unit_Price), des métriques transactionnelles (Quantity, Discount_Amount) et des données temporelles (Date). Cette richesse informationnelle permet une analyse multidimensionnelle approfondie du comportement client.

Le résumé statistique fournit des insights préliminaires importants. L'analyse des moyennes, médianes et écarts-types pour les variables numériques permet d'identifier la distribution des âges des clients, la gamme de prix des produits, les quantités typiquement commandées et l'ampleur des remises accordées. Les variables catégorielles, notamment les catégories de produits et le genre, montrent une distribution équilibrée, garantissant que l'analyse ne sera pas biaisée par une surreprésentation d'un segment particulier.

---

## ⚙️ 3. Ingénierie des Caractéristiques

### 3.1 Création de Variables Dérivées

```python
# Conversion de la date en format datetime pour permettre les manipulations temporelles
df['Date'] = pd.to_datetime(df['Date'])

# === VARIABLES FINANCIÈRES ===
# Calcul du montant total avant remise
df['Total_Amount'] = df['Unit_Price'] * df['Quantity']

# Calcul du montant final après application de la remise
df['Final_Amount'] = df['Total_Amount'] - df['Discount_Amount']

# Calcul du pourcentage de remise par rapport au total
df['Discount_Percentage'] = (df['Discount_Amount'] / df['Total_Amount'] * 100).fillna(0)

# === VARIABLES TEMPORELLES ===
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Day_Name'] = df['Date'].dt.day_name()
df['Quarter'] = df['Date'].dt.quarter
df['Week'] = df['Date'].dt.isocalendar().week

# === SEGMENTATION DÉMOGRAPHIQUE ===
df['Age_Group'] = pd.cut(df['Age'], 
                          bins=[0, 25, 35, 45, 55, 100],
                          labels=['18-25', '26-35', '36-45', '46-55', '55+'])

# === INDICATEURS BINAIRES ===
df['Has_Discount'] = (df['Discount_Amount'] > 0).astype(int)

print("✅ Ingénierie des caractéristiques terminée avec succès!")
print(f"🎯 Nombre total de variables: {df.shape[1]} colonnes")
print(f"\n🔧 Nouvelles variables créées:")

new_features = ['Total_Amount', 'Final_Amount', 'Discount_Percentage', 
                'Year', 'Month', 'Month_Name', 'Day', 'DayOfWeek', 
                'Day_Name', 'Quarter', 'Week', 'Age_Group', 'Has_Discount']
for i, feat in enumerate(new_features, 1):
    print(f"   {i}. {feat}")
```

**Interprétation :** L'ingénierie des caractéristiques constitue une étape cruciale qui transforme les données brutes en variables analytiquement exploitables. Cette phase enrichit considérablement le dataset en créant 13 nouvelles variables qui permettront des analyses plus approfondies et nuancées.

Les variables financières créées sont particulièrement importantes pour l'analyse de la rentabilité. Le `Total_Amount` représente la valeur brute des transactions, tandis que le `Final_Amount` reflète la valeur réelle encaissée après remises. Cette distinction est essentielle pour évaluer l'impact réel des promotions sur les revenus. Le `Discount_Percentage` permet de comparer l'intensité des remises indépendamment de la valeur absolue des transactions, facilitant ainsi les comparaisons entre différentes catégories de produits ou segments de clients.

Les variables temporelles décomposent la dimension temps en multiples facettes exploitables. L'extraction de l'année, du mois, du jour de la semaine, du trimestre et de la semaine permet d'identifier des patterns saisonniers, des tendances hebdomadaires et des variations mensuelles. Par exemple, l'analyse du jour de la semaine peut révéler que les clients achètent davantage en fin de semaine, tandis que l'analyse trimestrielle peut mettre en évidence des périodes de forte activité liées à des événements commerciaux spécifiques.

La segmentation par groupes d'âge transforme une variable continue en catégories interprétables, facilitant l'analyse des comportements générationnels. Cette segmentation permet d'identifier si certains groupes d'âge ont des préférences produits distinctes, des sensibilités différentes aux prix ou des habitudes d'achat spécifiques. L'indicateur binaire `Has_Discount` simplifie l'analyse comparative entre les transactions avec et sans promotion, permettant d'évaluer rapidement l'efficacité des stratégies de remise.

---

## 🛍️ 4. Analyse de la Performance des Catégories de Produits

### 4.1 Visualisations Complètes par Catégorie

```python
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. COMMANDES PAR CATÉGORIE (Barres horizontales)
ax1 = fig.add_subplot(gs[0, :2])
category_orders = df['Product_Category'].value_counts().sort_values(ascending=True)
colors_cat = plt.cm.Set3(np.linspace(0, 1, len(category_orders)))
bars1 = ax1.barh(category_orders.index, category_orders.values, color=colors_cat,
                 edgecolor='black', linewidth=1.5, alpha=0.85)
ax1.set_xlabel('Nombre de Commandes', fontsize=11, weight='bold')
ax1.set_title('📦 Volume de Commandes par Catégorie de Produits', 
              fontsize=13, weight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars1, category_orders.values)):
    ax1.text(val, i, f' {val:,}', va='center', fontsize=10, weight='bold')

# 2. DISTRIBUTION DES CATÉGORIES (Donut Chart)
ax2 = fig.add_subplot(gs[0, 2])
category_dist = df['Product_Category'].value_counts()
colors_donut = plt.cm.Pastel1(np.linspace(0, 1, len(category_dist)))
wedges, texts, autotexts = ax2.pie(category_dist.values, labels=category_dist.index,
                                     autopct='%1.1f%%', startangle=90, colors=colors_donut,
                                     textprops={'fontsize': 8, 'weight': 'bold'},
                                     pctdistance=0.85)
circle = plt.Circle((0, 0), 0.70, fc='white')
ax2.add_artist(circle)
ax2.set_title('🥧 Répartition du Mix Produits', fontsize=12, weight='bold', pad=10)

# 3. REVENUS PAR CATÉGORIE
ax3 = fig.add_subplot(gs[1, :2])
category_revenue = df.groupby('Product_Category')['Final_Amount'].sum().sort_values(ascending=False)
bars3 = ax3.bar(range(len(category_revenue)), category_revenue.values,
                color=plt.cm.viridis(np.linspace(0.2, 0.9, len(category_revenue))),
                edgecolor='black', linewidth=1.5, alpha=0.85)
ax3.set_xticks(range(len(category_revenue)))
ax3.set_xticklabels(category_revenue.index, rotation=45, ha='right', fontsize=10)
ax3.set_ylabel('Revenu Total (₺)', fontsize=11, weight='bold')
ax3.set_title('💰 Contribution aux Revenus par Catégorie', 
              fontsize=13, weight='bold', pad=15)
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars3, category_revenue.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'₺{val/1000:.0f}K', ha='center', va='bottom', fontsize=9, weight='bold')

# 4. PANIER MOYEN PAR CATÉGORIE
ax4 = fig.add_subplot(gs[1, 2])
category_aov = df.groupby('Product_Category')['Final_Amount'].mean().sort_values(ascending=False)
bars4 = ax4.barh(category_aov.index, category_aov.values,
                 color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(category_aov))),
                 edgecolor='black', linewidth=1.2, alpha=0.85)
ax4.set_xlabel('Panier Moyen (₺)', fontsize=10, weight='bold')
ax4.set_title('📊 Valeur Moyenne par Commande', fontsize=12, weight='bold', pad=10)
ax4.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars4, category_aov.values)):
    ax4.text(val, i, f' ₺{val:.0f}', va='center', fontsize=8, weight='bold')

# 5. HEATMAP DE PERFORMANCE GLOBALE
ax5 = fig.add_subplot(gs[2, :])
category_metrics = df.groupby('Product_Category').agg({
    'Order_ID': 'count',
    'Final_Amount': ['sum', 'mean'],
    'Quantity': 'sum',
    'Discount_Amount': 'mean'
}).round(0)
category_metrics.columns = ['Commandes', 'Revenu Total', 'Panier Moyen', 
                             'Unités Vendues', 'Remise Moyenne']

# Normalisation pour la heatmap
category_metrics_normalized = (category_metrics - category_metrics.min()) / \
                              (category_metrics.max() - category_metrics.min())

sns.heatmap(category_metrics_normalized.T, annot=category_metrics.T, fmt=',.0f',
            cmap='YlOrRd', linewidths=1, linecolor='black', 
            cbar_kws={'label': 'Score Normalisé (0-1)'},
            ax=ax5, annot_kws={'fontsize': 9, 'weight': 'bold'})
ax5.set_title('🔥 Tableau de Bord de Performance Multi-Critères', 
              fontsize=13, weight='bold', pad=15)
ax5.set_xlabel('Catégorie de Produit', fontsize=11, weight='bold')
ax5.set_ylabel('Indicateurs de Performance', fontsize=11, weight='bold')
ax5.tick_params(axis='x', rotation=45)

plt.suptitle('🛍️ ANALYSE COMPLÈTE DE LA PERFORMANCE PAR CATÉGORIE', 
             fontsize=18, weight='bold', y=0.998)
plt.show()

# Identification des catégories championnes
print("\n🏆 CATÉGORIES CHAMPIONNES PAR DIMENSION:")
print("="*60)
print(f"📦 Plus grand volume de commandes: {category_orders.index[-1]}")
print(f"   → {category_orders.values[-1]:,} commandes")
print(f"\n💰 Meilleur contributeur aux revenus: {category_revenue.index[0]}")
print(f"   → ₺{category_revenue.values[0]:,.0f} de revenus générés")
print(f"\n💎 Panier moyen le plus élevé: {category_aov.index[0]}")
print(f"   → ₺{category_aov.values[0]:.2f} par commande")
```

**Interprétation détaillée :** L'analyse de la performance des catégories de produits révèle des dynamiques commerciales complexes et fascinantes qui nécessitent une compréhension nuancée pour optimiser la stratégie produit de la plateforme.

Le graphique du volume de commandes met en évidence les catégories les plus populaires auprès des consommateurs. Cependant, il est crucial de comprendre que le volume ne se traduit pas automatiquement en rentabilité maximale. Une catégorie peut dominer en termes de nombre de transactions tout en générant un revenu moyen ou faible par commande. Cette distinction est particulièrement importante pour l'allocation des ressources marketing et logistiques.

La répartition du mix produits, visualisée par le diagramme circulaire, offre une perspective stratégique sur la diversification du portfolio. Une distribution trop concentrée sur quelques catégories expose l'entreprise à des risques significatifs : la dépendance excessive à une catégorie peut s'avérer problématique si les préférences des consommateurs évoluent ou si la concurrence s'intensifie. À l'inverse, une distribution trop fragmentée peut diluer les efforts marketing et compliquer la gestion des stocks. L'équilibre observé dans les données suggère une stratégie de diversification raisonnée.

L'analyse des revenus par catégorie révèle souvent des surprises. Il n'est pas rare de constater qu'une catégorie moins populaire en volume génère des revenus substantiels grâce à des prix unitaires élevés. Ces catégories "premium" représentent des opportunités stratégiques importantes : elles contribuent significativement à la rentabilité tout en nécessitant potentiellement moins d'efforts logistiques. Identifier et cultiver ces catégories à forte valeur ajoutée devrait être une priorité.

Le panier moyen par catégorie constitue un indicateur clé de la valeur perçue et du positionnement prix. Les catégories avec un panier moyen élevé peuvent justifier des investissements marketing plus importants, car chaque client acquis génère davantage de revenus. Inversement, les catégories à faible panier moyen doivent compenser par le volume ou par des stratégies de vente croisée et de montée en gamme.

La heatmap de performance multi-critères synthétise brillamment l'ensemble de ces dimensions en un seul visuel. Elle permet d'identifier rapidement les catégories "étoiles" qui excellent sur tous les critères, les catégories "vaches à lait" qui génèrent des revenus stables, les catégories "dilemmes" qui montrent du potentiel mais nécessitent des investissements, et les catégories "poids morts" qui sous-performent et nécessitent soit un repositionnement, soit un retrait progressif du catalogue.

---

## 📈 5. Analyse Temporelle des Tendances de Ventes

### 5.1 Évolution Mensuelle et Saisonnalité

```python
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

# Agrégation des données par mois
monthly_data = df.groupby(df['Date'].dt.to_period('M')).agg({
    'Final_Amount': 'sum',
    'Order_ID': 'count',
    'Customer_ID': 'nunique',
    'Discount_Amount': 'sum'
}).reset_index()
monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()
monthly_data['AOV'] = monthly_data['Final_Amount'] / monthly_data['Order_ID']

# 1. TENDANCE DES REVENUS MENSUELS AVEC LIGNE DE TENDANCE
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(monthly_data['Date'], monthly_data['Final_Amount']/1000,
         marker='o', linewidth=3, markersize=8, color='#FF6B6B', 
         label='Revenus Mensuels', zorder=3)
ax1.fill_between(monthly_data['Date'], monthly_data['Final_Amount']/1000,
                  alpha=0.3, color='#FF6B6B')
ax1.set_title('💰 Évolution des Revenus Mensuels et Tendance Générale', 
              fontsize=14, weight='bold', pad=15)
ax1.set_xlabel('Date', fontsize=11, weight='bold')
ax1.set_ylabel('Revenu (Milliers ₺)', fontsize=11, weight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')

# Ajout de la ligne de tendance polynomiale
z = np.polyfit(range(len(monthly_data)), monthly_data['Final_Amount']/1000, 1)
p = np.poly1d(z)
ax1.plot(monthly_data['Date'], p(range(len(monthly_data))),
         "--", color='darkred', linewidth=2.5, alpha=0.8, label='Tendance Linéaire')
ax1.legend(fontsize=11, loc='best')

# Annotation des points extrêmes
max_idx = monthly_data['Final_Amount'].idxmax()
min_idx = monthly_data['Final_Amount'].idxmin()
ax1.annotate(f"Maximum: ₺{monthly_data.loc[max_idx, 'Final_Amount']/1000:.0f}K",
             xy=(monthly_data.loc[max_idx, 'Date'], 
                 monthly_data.loc[max_idx, 'Final_Amount']/1000),
             xytext=(10, 15), textcoords='offset points', fontsize=10, weight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax1.annotate(f"Minimum: ₺{monthly_data.loc[min_idx, 'Final_Amount']/1000:.0f}K",
             xy=(monthly_data.loc[min_idx, 'Date'], 
                 monthly_data.loc[min_idx, 'Final_Amount']/1000),
             xytext=(10, -25), textcoords='offset points', fontsize=10, weight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

# 2. VOLUME DE COMMANDES MENSUELLES
ax2 = fig.add_subplot(gs[1, 0])
colors_monthly = plt.cm.viridis(np.linspace(0.3, 0.9, len(monthly_data)))
ax2.bar(monthly_data['Date'], monthly_data['Order_ID'],
        color=colors_monthly, edgecolor='black', linewidth=1.2, alpha=0.85)
ax2.set_title('📦 Volume de Commandes Mensuelles', 
              fontsize=13, weight='bold', pad=15)
ax2.set_xlabel('Date', fontsize=11, weight='bold')
ax2.set_ylabel('Nombre de Commandes', fontsize=11, weight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. ÉVOLUTION DU PANIER MOYEN
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(monthly_data['Date'], monthly_data['AOV'],
         marker='s', linewidth=2.5, markersize=7, color='#4ECDC4',
         label='Panier Moyen')
ax3.fill_between(monthly_data['Date'], monthly_data['AOV'], 
                  alpha=0.3, color='#4ECDC4')
ax3.set_title('💵 Évolution du Panier Moyen (AOV)', 
              fontsize=13, weight='bold', pad=15)
ax3.set_xlabel('Date', fontsize=11, weight='bold')
ax3.set_ylabel('Panier Moyen (₺)', fontsize=11, weight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.tick_params(axis='x', rotation=45)
ax3.legend()

# Ligne moyenne
ax3.axhline(monthly_data['AOV'].mean(), color='red', linestyle='--', 
            linewidth=2, alpha=0.7, label=f"Moyenne: ₺{monthly_data['AOV'].mean():.2f}")
ax3.legend()

# 4. COMPARAISON TRIMESTRIELLE
ax4 = fig.add_subplot(gs[2, 0])
quarterly_data = df.groupby(['Year', 'Quarter'])['Final_Amount'].sum().reset_index()
quarterly_data['Period'] = quarterly_data['Year'].astype(str) + '-Q' + \
                            quarterly_data['Quarter'].astype(str)
colors_quarterly = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#C7CEEA']
bars = ax4.bar(quarterly_data['Period'], quarterly_data['Final_Amount']/1000,
               color=colors_quarterly[:len(quarterly_data)],
               edgecolor='black', linewidth=1.5, alpha=0.85)
ax4.set_title('📊 Performance Trimestrielle des Revenus', 
              fontsize=13, weight='bold', pad=15)
ax4.set_xlabel('Trimestre', fontsize=11, weight='bold')
ax4.set_ylabel('Revenu (Milliers ₺)', fontsize=11, weight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

for bar, val in zip(bars, quarterly_data['Final_Amount']/1000):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'₺{val:.0f}K', ha='center', va='bottom', fontsize=9, weight='bold')

# 5. CROISSANCE DE LA BASE CLIENTS ACTIFS
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(monthly_data['Date'], monthly_data['Customer_ID'],
         marker='D', linewidth=2.5, markersize=7, color='#95E1D3',
         label='Clients Uniques')
ax5.fill_between(monthly_data['Date'], monthly_data['Customer_ID'], 
                  alpha=0.3, color='#95E1D3')
ax5.set_title('👥 Évolution des Clients Actifs Mensuels', 
              fontsize=13, weight='bold', pad=15)
ax5.set_xlabel('Date', fontsize=11, weight='bold')
ax5.set_ylabel('Nombre de Clients Uniques', fontsize=11, weight='bold')
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.tick_params(axis='x', rotation=45)
ax5.legend()

# 6. TABLEAU DE MÉTRIQUES DE CROISSANCE
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')

# Calcul des indicateurs de croissance
first_month_revenue = monthly_data['Final_Amount'].iloc[0]
last_month_revenue = monthly_data['Final_Amount'].iloc[-1]
revenue_growth = ((last_month_revenue - first_month_revenue) / first_month_revenue * 100)

first_month_orders = monthly_data['Order_ID'].iloc[0]
last_month_orders = monthly_data['Order_ID'].iloc[-1]
order_growth = ((last_month_orders - first_month_orders) / first_month_orders * 100)

first_month_aov = monthly_data['AOV'].iloc[0]
last_month_aov = monthly_data['AOV'].iloc[-1]
aov_growth = ((last_month_aov - first_month_aov) / first_month_aov * 100)

metrics_data = [
    ['Métrique', 'Début de Période', 'Fin de Période', 'Croissance (%)'],
    ['Revenus', f'₺{first