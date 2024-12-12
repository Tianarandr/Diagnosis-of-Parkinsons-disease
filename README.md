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

### Machine Learning Classique

Nous avons utilisé plusieurs algorithmes classiques de classification pour effectuer la tâche, tels que :
- Naïve Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)
  
Les performances de ces méthodes sont satisfaisantes avec des résultats allant jusqu'à 98% de précision, mais elles nécessitent un prétraitement manuel des données pour extraire des caractéristiques pertinentes.

### Réseaux de Neurones Convolutifs (CNN)

Afin de simplifier le processus et automatiser l'extraction des caractéristiques, nous avons implémenté un modèle basé sur un réseau de neurones convolutifs. Cette méthode, couramment utilisée dans la vision par ordinateur, a permis d'obtenir des résultats prometteurs, même avec une architecture simple et un jeu de données réduit. Cependant, des améliorations sont encore possibles, notamment en appliquant la méthode de fine-tuning pour affiner le modèle.

## 5. Résultats

Les résultats préliminaires montrent que les méthodes classiques de Machine Learning offrent des performances acceptables, mais l'utilisation des réseaux de neurones convolutifs permet de simplifier le processus et de potentiellement améliorer les résultats. Des améliorations supplémentaires sont à prévoir pour obtenir une meilleure précision, notamment avec plus de données et en optimisant les paramètres du modèle CNN.

## 6. Conclusion

Dans ce projet, nous avons exploré différentes approches pour diagnostiquer la maladie de Parkinson à partir de données de marche collectées à l'aide de capteurs. Nous avons montré qu'il est possible de classer les sujets en deux classes : parkinsonien et sain, avec une précision élevée grâce à des méthodes classiques de Machine Learning. L'utilisation des réseaux de neurones convolutifs permet d'automatiser l'extraction des caractéristiques et d'obtenir des résultats compétitifs avec des ressources limitées.

Cependant, des efforts supplémentaires sont nécessaires pour optimiser les modèles et améliorer la précision des classifications, en particulier en utilisant des techniques avancées comme le fine-tuning.

## 7. Installation

### Prérequis
- Python 3.x
- Bibliothèques nécessaires : `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`

### Installation
1. Clonez le repository :
   ```bash
   git clone https://github.com/Tianarandr/Diagnosis-of-Parkinsons-disease/
   cd Diagnosis-of-Parkinsons-disease
