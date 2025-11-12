
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

&lt;span style='color:#696969; '&gt;# Importation des bibliothèques&lt;/span&gt;
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

&lt;span style='color:#696969; '&gt;# Chargement de la base de données&lt;/span&gt;
&lt;span style='color:#696969; '&gt;# Remplacez 'Absenteeism_at_work.csv' par le chemin de ton fichier&lt;/span&gt;
df &lt;span style='color:#808030; '&gt;=&lt;/span&gt; pd.concat([X&lt;span style='color:#808030; '&gt;,&lt;/span&gt; y]&lt;span style='color:#808030; '&gt;,&lt;/span&gt; axis=&lt;span style='color:#008c00; '&gt;1&lt;/span&gt;)

&lt;span style='color:#696969; '&gt;# Aperçu des données&lt;/span&gt;
print(df.head())
print(df.info())

&lt;span style='color:#696969; '&gt;# -----------------------------&lt;/span&gt;
&lt;span style='color:#696969; '&gt;# 1. Histogramme de l'absentéisme (durée en heures)&lt;/span&gt;
plt.figure(figsize&lt;span style='color:#808030; '&gt;=&lt;/span&gt;(&lt;span style='color:#008c00; '&gt;8&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt;&lt;span style='color:#008c00; '&gt;5&lt;/span&gt;))
sns.histplot(df[&lt;span style='color:#0000e6; '&gt;'Absenteeism time in hours'&lt;/span&gt;]&lt;span style='color:#808030; '&gt;,&lt;/span&gt; bins&lt;span style='color:#808030; '&gt;=&lt;/span&gt;&lt;span style='color:#008c00; '&gt;30&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; kde=True&lt;span style='color:#808030; '&gt;,&lt;/span&gt; color=&lt;span style='color:#0000e6; '&gt;'skyblue'&lt;/span&gt;)
plt.title(&lt;span style='color:#0000e6; '&gt;'Distribution de la durée des absences (heures)'&lt;/span&gt;)
plt.xlabel(&lt;span style='color:#0000e6; '&gt;'Durée d\'&lt;/span&gt;absence (heures)&lt;span style='color:#0000e6; '&gt;')&lt;/span&gt;
plt.ylabel(&lt;span style='color:#0000e6; '&gt;'Nombre d\'&lt;/span&gt;absences&lt;span style='color:#0000e6; '&gt;')&lt;/span&gt;
plt.show()

&lt;span style='color:#696969; '&gt;# -----------------------------&lt;/span&gt;
&lt;span style='color:#696969; '&gt;# 2. Absences par raison (Reason for absence)&lt;/span&gt;
plt.figure(figsize&lt;span style='color:#808030; '&gt;=&lt;/span&gt;(&lt;span style='color:#008c00; '&gt;12&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt;&lt;span style='color:#008c00; '&gt;6&lt;/span&gt;))
sns.countplot(&lt;span style='color:#800000; font-weight:bold; '&gt;data&lt;/span&gt;&lt;span style='color:#808030; '&gt;=&lt;/span&gt;df&lt;span style='color:#808030; '&gt;,&lt;/span&gt; x=&lt;span style='color:#0000e6; '&gt;'Reason for absence'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; palette=&lt;span style='color:#0000e6; '&gt;'Set2'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; hue=&lt;span style='color:#0000e6; '&gt;'Reason for absence'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; legend=False)
plt.title(&lt;span style='color:#0000e6; '&gt;'Nombre d\'&lt;/span&gt;absences selon la raison&lt;span style='color:#0000e6; '&gt;')&lt;/span&gt;
plt.xlabel(&lt;span style='color:#0000e6; '&gt;'Raison de l\'&lt;/span&gt;absence&lt;span style='color:#0000e6; '&gt;')&lt;/span&gt;
plt.ylabel(&lt;span style='color:#0000e6; '&gt;'Nombre d\'&lt;/span&gt;absences&lt;span style='color:#0000e6; '&gt;')&lt;/span&gt;
plt.xticks(rotation&lt;span style='color:#808030; '&gt;=&lt;/span&gt;&lt;span style='color:#008c00; '&gt;45&lt;/span&gt;)
plt.show()

&lt;span style='color:#696969; '&gt;# -----------------------------&lt;/span&gt;
&lt;span style='color:#696969; '&gt;# 3. Absences par mois&lt;/span&gt;
plt.figure(figsize&lt;span style='color:#808030; '&gt;=&lt;/span&gt;(&lt;span style='color:#008c00; '&gt;10&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt;&lt;span style='color:#008c00; '&gt;5&lt;/span&gt;))
sns.countplot(&lt;span style='color:#800000; font-weight:bold; '&gt;data&lt;/span&gt;&lt;span style='color:#808030; '&gt;=&lt;/span&gt;df&lt;span style='color:#808030; '&gt;,&lt;/span&gt; x=&lt;span style='color:#0000e6; '&gt;'Month of absence'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; palette=&lt;span style='color:#0000e6; '&gt;'Set3'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; hue=&lt;span style='color:#0000e6; '&gt;'Month of absence'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; legend=False)
plt.title(&lt;span style='color:#0000e6; '&gt;'Répartition des absences par mois'&lt;/span&gt;)
plt.xlabel(&lt;span style='color:#0000e6; '&gt;'Mois'&lt;/span&gt;)
plt.ylabel(&lt;span style='color:#0000e6; '&gt;'Nombre d\'&lt;/span&gt;absences&lt;span style='color:#0000e6; '&gt;')&lt;/span&gt;
plt.show()

&lt;span style='color:#696969; '&gt;# -----------------------------&lt;/span&gt;
&lt;span style='color:#696969; '&gt;# 4. Boxplot de l'absentéisme par âge&lt;/span&gt;
plt.figure(figsize&lt;span style='color:#808030; '&gt;=&lt;/span&gt;(&lt;span style='color:#008c00; '&gt;10&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt;&lt;span style='color:#008c00; '&gt;5&lt;/span&gt;))
sns.boxplot(x&lt;span style='color:#808030; '&gt;=&lt;/span&gt;&lt;span style='color:#0000e6; '&gt;'Age'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; y=&lt;span style='color:#0000e6; '&gt;'Absenteeism time in hours'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; data=df&lt;span style='color:#808030; '&gt;,&lt;/span&gt; palette=&lt;span style='color:#0000e6; '&gt;'Pastel1'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; hue=&lt;span style='color:#0000e6; '&gt;'Age'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; legend=False)
plt.title(&lt;span style='color:#0000e6; '&gt;'Durée des absences selon l\'&lt;/span&gt;âge&lt;span style='color:#0000e6; '&gt;')&lt;/span&gt;
plt.xlabel(&lt;span style='color:#0000e6; '&gt;'Âge'&lt;/span&gt;)
plt.ylabel(&lt;span style='color:#0000e6; '&gt;'Durée d\'&lt;/span&gt;absence (heures)&lt;span style='color:#0000e6; '&gt;')&lt;/span&gt;
plt.show()

&lt;span style='color:#696969; '&gt;# -----------------------------&lt;/span&gt;
&lt;span style='color:#696969; '&gt;# 5. Corrélation entre variables numériques&lt;/span&gt;
plt.figure(figsize&lt;span style='color:#808030; '&gt;=&lt;/span&gt;(&lt;span style='color:#008c00; '&gt;12&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt;&lt;span style='color:#008c00; '&gt;8&lt;/span&gt;))
numeric_cols &lt;span style='color:#808030; '&gt;=&lt;/span&gt; [&lt;span style='color:#0000e6; '&gt;'Transportation expense'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; &lt;span style='color:#0000e6; '&gt;'Distance from Residence to Work'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt;
                &lt;span style='color:#0000e6; '&gt;'Service time'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; &lt;span style='color:#0000e6; '&gt;'Age'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; &lt;span style='color:#0000e6; '&gt;'Work load Average/day '&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; &lt;span style='color:#0000e6; '&gt;'Hit target'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt;
                &lt;span style='color:#0000e6; '&gt;'Absenteeism time in hours'&lt;/span&gt;]
corr &lt;span style='color:#808030; '&gt;=&lt;/span&gt; df[numeric_cols].corr()
sns.heatmap(corr&lt;span style='color:#808030; '&gt;,&lt;/span&gt; annot&lt;span style='color:#808030; '&gt;=&lt;/span&gt;True&lt;span style='color:#808030; '&gt;,&lt;/span&gt; cmap=&lt;span style='color:#0000e6; '&gt;'coolwarm'&lt;/span&gt;&lt;span style='color:#808030; '&gt;,&lt;/span&gt; fmt=&lt;span style='color:#0000e6; '&gt;".2f"&lt;/span&gt;)
plt.title(&lt;span style='color:#0000e6; '&gt;'Matrice de corrélation des variables numériques'&lt;/span&gt;)
plt.show()
&lt;!--Created using ToHTML.com on 2025-11-12 15:36:54 UTC --&gt;

## Graphiques 

  <img src="Graphique ASS1.png" style="height:400px;margin-right:250px"/>
  <img src="Graphique ASS2.png" style="height:400px;margin-right:250px"/> 
  <img src="Graphique ASS3.png" style="height:400px;margin-right:250px"/>
  <img src="Graphique ASS4.png" style="height:400px;margin-right:250px"/>
  <img src="Graphique ASS5.png" style="height:400px;margin-right:250px"/>

### 5. Intérêt analytique et applications
Cette base permet de réaliser plusieurs analyses :
- **Analyse descriptive** pour observer la répartition des absences selon l’âge, la saison ou la distance au travail.
- **Analyse prédictive** avec des modèles de machine learning afin d’anticiper les absences futures.
- **Analyse explicative** pour déterminer les facteurs ayant le plus d’impact sur le comportement d’absentéisme.

Les résultats peuvent aider les entreprises à identifier les causes principales de l’absentéisme, améliorer les politiques de gestion du personnel et mettre en place des actions de prévention.

### 6. Conclusion
La base de données “Absenteeism at Work” constitue un **outil précieux d’analyse et d’apprentissage** dans le domaine de la **gestion des ressources humaines**. Elle permet de comprendre les comportements d’absence des employés et de développer des modèles prédictifs utiles pour optimiser l’organisation interne. Elle illustre parfaitement l’usage de la **science des données** pour résoudre des problématiques managériales réelles.
