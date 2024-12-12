# Diagnostic de la maladie de Parkinson à l'aide des réseaux de neurones artificiels

## 1. Introduction

La maladie de Parkinson est une maladie neurodégénérative caractérisée par la destruction d’une population spécifique de neurones, et elle est la deuxième maladie neurodégénérative la plus fréquente en France, après la maladie d'Alzheimer. Cette pathologie constitue une cause majeure de handicap chez le sujet âgé. Les traitements actuels permettent de contrôler les symptômes moteurs associés à la maladie, mais ils sont sans effet sur les autres symptômes et sur la progression de la dégénérescence. 

Les chercheurs développent des stratégies neuroprotectrices visant à enrayer cette progression et cherchent également à repérer les patients à risque pour leur administrer ces traitements dès que possible.

Ce travail vise à diagnostiquer la maladie de Parkinson, même dans ses stades les moins sévères, en utilisant des techniques avancées de Machine Learning et des réseaux de neurones artificiels. Le projet se base sur la base de données "Gait in Parkinson's Disease 1.0.0", qui contient des enregistrements effectués sur des sujets parkinsoniens à différents stades de la maladie. 

Le projet utilise dans un premier temps des méthodes classiques de Machine Learning, puis des techniques plus récentes comme les réseaux de neurones convolutifs (CNN), souvent utilisées dans le domaine de la vision par ordinateur.

## 2. Description de la base de données

La base de données est composée de 166 échantillons représentant 93 sujets atteints de la maladie de Parkinson et 73 sujets sains. Chaque enregistrement comprend deux types de données :
- **Données spatio-temporelles** : Mesures de la force de réaction verticale au sol (VGRF) collectées à l'aide d'un système d'acquisition de données.
- **Données anthropométriques** : Informations sur les caractéristiques physiques des sujets.

Les données temporelles sont collectées grâce à huit capteurs sous chaque pied d'un sujet, enregistrant les forces de réaction pendant que celui-ci marche sur un terrain plat. Les enregistrements sont effectués pendant environ 2 minutes et les données sont ensuite transférées à un ordinateur pour analyse.

### Schéma de la base de données
<img width="520" alt="Capture d’écran 2024-12-12 à 21 26 33" src="https://github.com/user-attachments/assets/f8039e8c-b5c9-4e63-8a1b-016daff6a3f5" />



### Description des capteurs

Les capteurs sont placés sous chaque pied du sujet, mesurant la force de réaction au sol pendant la marche. La configuration du système de collecte est illustrée dans la figure suivante :


## 3. Objectifs du projet

- Utiliser des méthodes classiques de Machine Learning (Naïve Bayes, Decision Tree, etc.) pour la classification des sujets.
- Implémenter des réseaux de neurones convolutifs (CNN) pour automatiser l'extraction des caractéristiques pertinentes des signaux et améliorer les résultats de classification.
- Comparer les performances des différentes méthodes et présenter les résultats obtenus.

## 4. Méthodes utilisées

### Apprentissage Classique

Pour cette partie, nous avons choisi les modèles suivants :
- **Naïve Bayes (NB)** : Utilisation d’une distribution normale pour les trois sous-ensembles de données.
- **Arbres de décision (DT)** : L'algorithme CART est utilisé avec l'indice de Gini pour trouver la meilleure construction et la meilleure partition de l'arbre.
- **Forêts aléatoires (RF)** : Le nombre optimal d'arbres pour les trois sous-ensembles de données est fixé à 100.
- **Support Machines vectorielles (SVM)** : Modèle non linéaire avec noyau « rbf ».
- **Modèles de mélange gaussien (GMM)** : Nombre de classes fixé à 2 (parkinsonien et sain).
- **K-means** : Le seul paramètre à régler est le nombre de classes, qui est fixé à deux (sains ou malades).
- **KNN (K-Nearest Neighbors)** : Utilisation de la distance euclidienne, avec un nombre de voisins fixé à 3.

### Apprentissage Profond

Voici un tableau récapitulatif des résultats obtenus pour la partie d’apprentissage profond avec les différentes architectures testées :

| Modèle                        | Frenkel Toledo et al | Galit Yogev et al | Hausdoff et al | Accuracy (%) | Loss (%) |
|-------------------------------|----------------------|-------------------|----------------|--------------|----------|
| **Perceptrons multicouches**   | 80                   | 75                | 83.3           | 36           | 44       | 31 |
| **CNN (Convolutional Neural Network)**  | 86.6                 | 82.1              | 87.0           | 27           | 36       | 25.96    |
| **FCN (Fully Convolutional Network)**   | 91                   | 86.3              | 80             | 13           | 21       | 33       |
| **ResNet**                     | 93.3                 | 81                | 89.85          | 9.8          | 15.9     | 21.7     |

**Tableau 2 : Résultats des modèles d’apprentissage automatique profond**

## 5. Performance des modèles

### Modèles supervisés vs non supervisés

Voici un tableau détaillant les performances des différents modèles utilisés dans l'apprentissage supervisé et non supervisé, en termes de précision, rappel, F-mesure et écart type (STD) :

| Modèle      | Accuracy (%) | STD (%) | Précision (%) | Recall (%) | F-Measure (%) |
|-------------|--------------|---------|---------------|------------|---------------|
| **KNN**     | 94.7         | 2.61    | 86.13         | 71.72      | 81.38         |
| **SVM**     | 86.61        | 3.49    | 83.74         | 67.91      | 79.33         |
| **RF**      | 89.27        | 5.61    | 86.64         | 61.97      | 73.38         |
| **DT**      | 84.86        | 3.83    | 77.99         | 57.30      | 66.71         |
| **NB**      | 80.21        | 4.60    | 75.68         | 64.43      | 68.90         |
| **GMM**     | 70.13        | 5.61    | 37.14         | 27.80      | 29.94         |
| **K-Means** | 65.34        | 3.83    | 38.17         | 49.91      | 30.19         |

**Tableau 3 : Performance des modèles supervisés et non supervisés**

## 6. Conclusion

Les résultats montrent que les modèles supervisés, comme KNN, SVM et RF, offrent des performances de classification nettement supérieures par rapport aux modèles non supervisés tels que GMM et K-Means. En particulier, KNN et SVM atteignent les meilleures performances en termes d'accuracy, précision et F-mesure. Les modèles non supervisés, bien qu'intéressants pour l'exploration des données, semblent moins performants pour cette tâche de classification spécifique.

## 7. Installation

### Prérequis
- Python 3.x
- Bibliothèques nécessaires : `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`


### Installation
1. Clonez le repository :
   ```bash
   git clone https://github.com/Tianarandr/Diagnosis-of-Parkinsons-disease/
   cd Diagnosis-of-Parkinsons-disease
