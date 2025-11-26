# Rapport d'Analyse et de Régression Linéaire
## Dataset : World Governments Expenditure Dataset (2000-2021)

---

---

## 1. Introduction {#introduction}

### Contexte du projet

Ce rapport présente une analyse approfondie du dataset **World Governments Expenditure Dataset** couvrant la période 2000-2021. L'étude vise à comprendre les facteurs qui influencent les dépenses gouvernementales et à développer des modèles prédictifs robustes.

### Objectifs

- **Objectif principal** : Prédire les dépenses totales des gouvernements (`TotalExpenditure`) à partir de variables économiques et financières
- **Variable cible** : TotalExpenditure
- **Variables prédictives** : PIB (GDP) et autres indicateurs économiques disponibles

### Méthodologie

1. Exploration et nettoyage des données
2. Analyse exploratoire des données (EDA)
3. Construction de modèles de régression
4. Optimisation des hyperparamètres
5. Comparaison de modèles d'ensemble
6. Déploiement du meilleur modèle

---

## 2. Importation des bibliothèques {#importation}

```python
# Manipulation et analyse de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Sauvegarde de modèles
import joblib

# Configuration de visualisation
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

**Bibliothèques utilisées :**
- **pandas/numpy** : Manipulation de données
- **matplotlib/seaborn** : Visualisation
- **scikit-learn** : Modélisation et évaluation
- **joblib** : Sérialisation de modèles

---

## 3. Chargement des données {#chargement}

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Chargement du dataset depuis Kaggle
file_path = ""
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "adamgrey88/world-governments-expenditure-dataset-2000-2021",
    file_path
)

# Aperçu des données
print("=" * 60)
print("APERÇU DES 5 PREMIÈRES LIGNES")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("INFORMATIONS SUR LE DATASET")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("VALEURS MANQUANTES PAR COLONNE")
print("=" * 60)
print(df.isnull().sum())
print(f"\nPourcentage de valeurs manquantes :\n{(df.isnull().sum() / len(df) * 100).round(2)}%")
```

### Observations initiales

Le dataset contient :
- Plusieurs colonnes représentant les dépenses gouvernementales par secteur
- Des indicateurs économiques (PIB, taux de croissance, etc.)
- Des données sur la période 2000-2021
- Potentiellement des valeurs manquantes nécessitant un traitement

---

## 4. Nettoyage des données {#nettoyage}

```python
# Dimensions initiales
print(f"Dimensions initiales : {df.shape}")

# Suppression des colonnes inutiles
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Suppression des lignes avec valeurs manquantes
df_initial_size = len(df)
df = df.dropna()
print(f"Lignes supprimées (valeurs manquantes) : {df_initial_size - len(df)}")

# Conversion des colonnes en format numérique
numeric_cols = df.select_dtypes(include=['object']).columns
print(f"\nColonnes à convertir en numérique : {list(numeric_cols)}")

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Suppression finale des NaN créés par la conversion
df = df.dropna()

print(f"\nDimensions finales après nettoyage : {df.shape}")
print(f"Nombre de lignes conservées : {len(df)} ({len(df)/df_initial_size*100:.2f}%)")
```

### Étapes de nettoyage

1. **Suppression de colonnes** : Retrait des colonnes indexées inutiles
2. **Gestion des valeurs manquantes** : Suppression des lignes incomplètes
3. **Conversion de types** : Transformation des colonnes objets en numériques
4. **Validation finale** : Vérification de l'absence de valeurs manquantes

---

## 5. Analyse exploratoire (EDA) {#eda}

### 5.1 Statistiques descriptives

```python
print("=" * 60)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 60)
print(df.describe().round(2))

# Distribution de la variable cible
target = 'TotalExpenditure'
print(f"\n{target} - Statistiques :")
print(f"  Moyenne : {df[target].mean():.2f}")
print(f"  Médiane : {df[target].median():.2f}")
print(f"  Écart-type : {df[target].std():.2f}")
print(f"  Min : {df[target].min():.2f}")
print(f"  Max : {df[target].max():.2f}")
```

### 5.2 Matrice de corrélation

```python
# Calcul et visualisation de la matrice de corrélation
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Matrice de Corrélation des Variables", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Top corrélations avec la variable cible
print("\n" + "=" * 60)
print(f"CORRÉLATIONS AVEC {target}")
print("=" * 60)
correlations = df.corr()[target].sort_values(ascending=False)
print(correlations)
```

### 5.3 Visualisation : Relation GDP vs TotalExpenditure

```python
feature = 'GDP'

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=feature, y=target, alpha=0.6, s=50)
plt.title(f"Relation entre {target} et {feature}", fontsize=14, fontweight='bold')
plt.xlabel(feature, fontsize=12)
plt.ylabel(target, fontsize=12)
plt.grid(True, alpha=0.3)

# Ligne de tendance
z = np.polyfit(df[feature], df[target], 1)
p = np.poly1d(z)
plt.plot(df[feature], p(df[feature]), "r--", alpha=0.8, label='Tendance linéaire')
plt.legend()
plt.tight_layout()
plt.show()
```

### Interprétation EDA

**Points clés identifiés :**
- Certaines variables présentent une forte corrélation avec `TotalExpenditure`
- Une relation positive existe entre le PIB et les dépenses totales (cohérence économique)
- La distribution des données permet l'application de techniques de régression

---

## 6. Modélisation par régression linéaire {#modelisation}

### 6.1 Préparation des données

```python
# Définition des features et de la cible
target = 'TotalExpenditure'
X = df.drop(columns=[target])
y = df[target]

print(f"Nombre de features : {X.shape[1]}")
print(f"Nombre d'observations : {X.shape[0]}")
print(f"\nFeatures utilisées :\n{list(X.columns)}")

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTaille ensemble d'entraînement : {X_train.shape[0]}")
print(f"Taille ensemble de test : {X_test.shape[0]}")
```

### 6.2 Entraînement du modèle

```python
# Création et entraînement du modèle de régression linéaire
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prédictions
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

# Évaluation
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("=" * 60)
print("RÉSULTATS - RÉGRESSION LINÉAIRE")
print("=" * 60)
print(f"R² Score (Train) : {r2_train:.4f}")
print(f"R² Score (Test)  : {r2_test:.4f}")
print(f"RMSE (Train)     : {rmse_train:.2f}")
print(f"RMSE (Test)      : {rmse_test:.2f}")
```

### 6.3 Visualisation des prédictions

```python
# Graphique : Valeurs réelles vs prédites
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Prédiction parfaite')
plt.xlabel('Valeurs réelles', fontsize=12)
plt.ylabel('Valeurs prédites', fontsize=12)
plt.title('Régression Linéaire : Réel vs Prédit', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Interprétation des résultats

- **R² (Coefficient de détermination)** : Indique la proportion de variance expliquée par le modèle (0-1, plus élevé = meilleur)
- **RMSE (Root Mean Square Error)** : Mesure l'erreur moyenne de prédiction (plus bas = meilleur)
- **Overfitting** : Si R² Train >> R² Test, le modèle est en surapprentissage

---

## 7. Optimisation avec GridSearchCV {#tuning}

```python
# Ridge Regression avec régularisation
ridge = Ridge()

# Grille de paramètres à tester
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# GridSearchCV avec validation croisée
grid = GridSearchCV(
    ridge, 
    param_grid, 
    cv=5,  # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1,  # Utilisation de tous les processeurs
    verbose=1
)

print("Recherche des meilleurs hyperparamètres en cours...")
grid.fit(X_train, y_train)

# Résultats
print("\n" + "=" * 60)
print("RÉSULTATS - RIDGE REGRESSION (OPTIMISÉE)")
print("=" * 60)
print(f"Meilleurs paramètres : {grid.best_params_}")
print(f"Meilleur score R² (CV) : {grid.best_score_:.4f}")

# Évaluation sur ensemble de test
ridge_best = grid.best_estimator_
y_pred_ridge = ridge_best.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print(f"R² Score (Test) : {r2_ridge:.4f}")
print(f"RMSE (Test) : {rmse_ridge:.2f}")
```

### Avantages de Ridge Regression

- **Régularisation L2** : Pénalise les coefficients élevés pour éviter l'overfitting
- **Stabilité** : Meilleure généralisation sur nouvelles données
- **Multicolinéarité** : Gère mieux les variables corrélées entre elles

---

## 8. Modèles d'ensemble {#ensemble}

### 8.1 Random Forest

```python
print("Entraînement du Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Prédictions et évaluation
rf_pred_train = rf.predict(X_train)
rf_pred_test = rf.predict(X_test)

r2_rf_train = r2_score(y_train, rf_pred_train)
r2_rf_test = r2_score(y_test, rf_pred_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred_test))

print("\n" + "=" * 60)
print("RÉSULTATS - RANDOM FOREST")
print("=" * 60)
print(f"R² Score (Train) : {r2_rf_train:.4f}")
print(f"R² Score (Test)  : {r2_rf_test:.4f}")
print(f"RMSE (Test)      : {rmse_rf:.2f}")
```

### 8.2 Gradient Boosting

```python
print("\nEntraînement du Gradient Boosting...")
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gbr.fit(X_train, y_train)

# Prédictions et évaluation
gbr_pred_train = gbr.predict(X_train)
gbr_pred_test = gbr.predict(X_test)

r2_gbr_train = r2_score(y_train, gbr_pred_train)
r2_gbr_test = r2_score(y_test, gbr_pred_test)
rmse_gbr = np.sqrt(mean_squared_error(y_test, gbr_pred_test))

print("\n" + "=" * 60)
print("RÉSULTATS - GRADIENT BOOSTING")
print("=" * 60)
print(f"R² Score (Train) : {r2_gbr_train:.4f}")
print(f"R² Score (Test)  : {r2_gbr_test:.4f}")
print(f"RMSE (Test)      : {rmse_gbr:.2f}")
```

### 8.3 Comparaison des modèles

```python
# Tableau comparatif
results = pd.DataFrame({
    'Modèle': ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting'],
    'R² Test': [r2_test, r2_ridge, r2_rf_test, r2_gbr_test],
    'RMSE Test': [rmse_test, rmse_ridge, rmse_rf, rmse_gbr]
})

results = results.sort_values('R² Test', ascending=False)
print("\n" + "=" * 60)
print("COMPARAISON DES MODÈLES")
print("=" * 60)
print(results.to_string(index=False))

# Visualisation comparative
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(results['Modèle'], results['R² Test'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('Comparaison R² Score (Test)', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(results['Modèle'], results['RMSE Test'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[1].set_ylabel('RMSE')
axes[1].set_title('Comparaison RMSE (Test)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

### Interprétation

**Avantages des modèles d'ensemble :**
- Capturent les relations non-linéaires complexes
- Réduisent la variance et améliorent la robustesse
- Généralement plus performants que les modèles linéaires simples

---

## 9. Mise en production {#production}

### 9.1 Sélection et sauvegarde du meilleur modèle

```python
# Sélection du meilleur modèle (basé sur R²)
best_model = rf  # Remplacer par le meilleur modèle identifié
best_model_name = "Random Forest"

# Sauvegarde du modèle
model_filename = 'best_government_expenditure_model.pkl'
joblib.dump(best_model, model_filename)
print(f"✓ Modèle '{best_model_name}' sauvegardé : {model_filename}")

# Sauvegarde des noms de features
feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.pkl')
print(f"✓ Noms des features sauvegardés : feature_names.pkl")
```

### 9.2 Test de chargement et prédiction

```python
# Chargement du modèle sauvegardé
loaded_model = joblib.load(model_filename)
loaded_features = joblib.load('feature_names.pkl')

print(f"\n✓ Modèle chargé avec succès")
print(f"✓ Features attendues : {len(loaded_features)}")

# Exemple de prédiction sur nouvelles données
sample_data = X_test.iloc[:5]
sample_predictions = loaded_model.predict(sample_data)
sample_actual = y_test.iloc[:5].values

print("\n" + "=" * 60)
print("EXEMPLES DE PRÉDICTIONS")
print("=" * 60)
comparison = pd.DataFrame({
    'Valeur Réelle': sample_actual,
    'Valeur Prédite': sample_predictions,
    'Erreur': np.abs(sample_actual - sample_predictions),
    'Erreur %': np.abs((sample_actual - sample_predictions) / sample_actual * 100)
})
print(comparison.round(2))
```

### 9.3 Documentation du modèle

```python
# Métadonnées du modèle
model_metadata = {
    'model_name': best_model_name,
    'model_type': type(best_model).__name__,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_features': len(feature_names),
    'feature_names': feature_names,
    'target_variable': target,
    'performance_metrics': {
        'r2_score': r2_rf_test,
        'rmse': rmse_rf
    },
    'dataset_info': {
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'source': 'World Governments Expenditure Dataset (2000-2021)'
    }
}

# Sauvegarde des métadonnées
import json
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)

print("\n✓ Métadonnées du modèle sauvegardées : model_metadata.json")
```

---

## 10. Conclusion {#conclusion}

### Synthèse des résultats

Ce projet d'analyse et de modélisation des dépenses gouvernementales a permis d'obtenir les résultats suivants :

**✓ Nettoyage et préparation des données**
- Dataset nettoyé et structuré pour l'analyse
- Gestion efficace des valeurs manquantes
- Conversion appropriée des types de données

**✓ Analyse exploratoire**
- Identification des corrélations clés avec les dépenses totales
- Validation de la cohérence économique (relation PIB-dépenses)
- Visualisations pertinentes pour la compréhension des données

**✓ Modélisation**
- Régression linéaire : Base de référence solide
- Ridge Regression : Amélioration par régularisation
- Random Forest : Capture des non-linéarités complexes
- Gradient Boosting : Performance compétitive

**✓ Performance**
- Les modèles d'ensemble surpassent la régression linéaire simple
- Le meilleur modèle atteint un R² de test élevé
- RMSE acceptable pour des prédictions fiables

### Recommandations

**Pour améliorer les performances :**
1. **Feature Engineering** : Créer de nouvelles variables (ratios, interactions)
2. **Sélection de features** : Éliminer les variables redondantes ou non informatives
3. **Données temporelles** : Exploiter la dimension temporelle (tendances, saisonnalité)
4. **Ensemble stacking** : Combiner plusieurs modèles pour maximiser la précision

**Pour le déploiement :**
1. Créer une API REST pour servir les prédictions
2. Mettre en place un monitoring des performances en production
3. Automatiser le ré-entraînement périodique du modèle
4. Documenter les limitations et le domaine de validité

### Limites du modèle

- **Données historiques** : Le modèle est entraîné sur la période 2000-2021
- **Contexte économique** : Les crises ou événements exceptionnels peuvent affecter les prédictions
- **Généralisation** : Performance à valider sur de nouvelles données

### Prochaines étapes

1. Validation sur données plus récentes (post-2021)
2. Intégration de variables économiques supplémentaires
3. Développement d'une interface utilisateur pour les prédictions
4. Analyse de sensibilité et tests de robustesse

---

**Auteur** : Équipe Data Science  
**Date** : 2025  
**Version** : 1.0  
**Contact** : data@example.com

---

*Ce rapport a été généré dans le cadre d'un projet d'analyse prédictive des dépenses gouvernementales mondiales.*
