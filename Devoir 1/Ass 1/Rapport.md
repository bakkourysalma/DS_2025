<img src="SETTAT.png" style="height:100px;margin-right:95px"/>  


## Thème du projet :
**Analyse des facteurs d’absentéisme au travail à partir de la base de données “Absenteeism at Work”**

## Rapport sur la base de données “Absenteeism at Work”
# BAKKOURY Salma 

<img src="Photo salma.jpg" style="height:200px;margin-right:150px"/>

 # Étudiante en 4ᵉ année à l’ENCG Settat – filière Contrôle, Audit et Conseil .
### 1. Introduction
L’absentéisme au travail est un phénomène qui touche la plupart des entreprises et représente un véritable défi pour les responsables des ressources humaines. Il peut avoir des conséquences importantes sur la productivité, la qualité du service et le climat social au sein de l’organisation. Afin de mieux comprendre les causes de ce phénomène et d’en proposer une analyse objective, une base de données intitulée **“Absenteeism at Work”** a été constituée dans une entreprise de transport au **Brésil**, couvrant la période de **juillet 2007 à juillet 2010**.

### 2. Contexte et origine
Cette base de données a été créée dans le cadre d’un projet de recherche mené par **Dr. José A. B. Fernandes** et **Ricardo P. de Sá**, chercheurs à l’**Université d’État de Campinas (UNICAMP)**, au Brésil. Leur objectif était d’étudier les **facteurs influençant l’absentéisme** au sein d’une entreprise et de fournir une base exploitable pour les analyses statistiques et prédictives. Les données ont ensuite été mises à disposition du public sur le **UCI Machine Learning Repository**.

### 3. Objectifs de l’étude
L’étude vise à **identifier et analyser les principaux déterminants de l’absentéisme** au travail. Elle cherche à répondre aux questions suivantes :
- Quels sont les **facteurs personnels, professionnels ou environnementaux** qui influencent la fréquence et la durée des absences ?
- Comment ces variables peuvent-elles aider à **prévoir** les absences futures ?
- De quelle manière les entreprises peuvent-elles **réduire l’absentéisme** et **améliorer la gestion de leurs ressources humaines** ?

### 4. Description du jeu de données
Le jeu de données comprend environ **740 enregistrements** et **21 variables**. Chaque observation représente un **épisode d’absence** d’un employé, avec :
- **Variables personnelles :** âge, poids, taille, IMC, nombre d’enfants, animaux de compagnie.
- **Variables professionnelles :** ancienneté, niveau d’éducation, performance, fautes disciplinaires, charge de travail moyenne.
- **Variables logistiques et temporelles :** mois, saison, jour de la semaine, distance domicile-travail, dépenses de transport.
- **Variable cible :** *Absenteeism time in hours*, durée totale de l’absence.
- **Autres variables :** raison de l’absence (codée selon une classification médicale).
- 
### Code python  
```python
# Importation des bibliothèques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement de la base de données
# Remplacez 'Absenteeism_at_work.csv' par le chemin de ton fichier
df = pd.concat([X, y], axis=1)

# Aperçu des données
print(df.head())
print(df.info())

# -----------------------------
# 1. Histogramme de l'absentéisme (durée en heures)
plt.figure(figsize=(8,5))
sns.histplot(df['Absenteeism time in hours'], bins=30, kde=True, color='skyblue')
plt.title('Distribution de la durée des absences (heures)')
plt.xlabel('Durée d\'absence (heures)')
plt.ylabel('Nombre d\'absences')
plt.show()

# -----------------------------
# 2. Absences par raison (Reason for absence)
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Reason for absence', palette='Set2', hue='Reason for absence', legend=False)
plt.title('Nombre d\'absences selon la raison')
plt.xlabel('Raison de l\'absence')
plt.ylabel('Nombre d\'absences')
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 3. Absences par mois
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Month of absence', palette='Set3', hue='Month of absence', legend=False)
plt.title('Répartition des absences par mois')
plt.xlabel('Mois')
plt.ylabel('Nombre d\'absences')
plt.show()

# -----------------------------
# 4. Boxplot de l'absentéisme par âge
plt.figure(figsize=(10,5))
sns.boxplot(x='Age', y='Absenteeism time in hours', data=df, palette='Pastel1', hue='Age', legend=False)
plt.title('Durée des absences selon l\'âge')
plt.xlabel('Âge')
plt.ylabel('Durée d\'absence (heures)')
plt.show()

# -----------------------------
# 5. Corrélation entre variables numériques
plt.figure(figsize=(12,8))
numeric_cols = ['Transportation expense', 'Distance from Residence to Work',
                'Service time', 'Age', 'Work load Average/day ', 'Hit target',
                'Absenteeism time in hours']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation des variables numériques')
plt.show()
```

## Graphiques 

  <img src="Graphique ASS1.png" style="height:500px;margin-right:350px"/>
  <img src="Graphique ASS2.png" style="height:500px;margin-right:350px"/> 
  <img src="Graphique ASS3.png" style="height:500px;margin-right:350px"/>
  <img src="Graphique ASS4.png" style="height:500px;margin-right:350px"/>
  <img src="Graphique ASS5.png" style="height:500px;margin-right:350px"/>

### 5. Intérêt analytique et applications
Cette base permet de réaliser plusieurs analyses :
- **Analyse descriptive** pour observer la répartition des absences selon l’âge, la saison ou la distance au travail.
- **Analyse prédictive** avec des modèles de machine learning afin d’anticiper les absences futures.
- **Analyse explicative** pour déterminer les facteurs ayant le plus d’impact sur le comportement d’absentéisme.

Les résultats peuvent aider les entreprises à identifier les causes principales de l’absentéisme, améliorer les politiques de gestion du personnel et mettre en place des actions de prévention.

### 6. Conclusion
La base de données “Absenteeism at Work” constitue un **outil précieux d’analyse et d’apprentissage** dans le domaine de la **gestion des ressources humaines**. Elle permet de comprendre les comportements d’absence des employés et de développer des modèles prédictifs utiles pour optimiser l’organisation interne. Elle illustre parfaitement l’usage de la **science des données** pour résoudre des problématiques managériales réelles.
