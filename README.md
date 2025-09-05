# Music Genre Classification using Machine Learning

## Project Overview
This project demonstrates the application of machine learning techniques to classify music genres using Spotify audio features. It implements supervised learning models (Support Vector Machines and Random Forest) to predict music genres based on acoustic properties. The project showcases a complete data science workflow including data auditing, preprocessing, feature engineering, model training, and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Technologies Used](#technologies-used)

## Introduction
Music genre classification is a fundamental problem in music information retrieval. This project leverages Spotify's audio features to develop machine learning models that can automatically classify tracks into different music genres. By analyzing acoustic properties like tempo, energy, and danceability, the models learn to distinguish between different musical styles.

## Dataset
The project uses a dataset containing Spotify audio features for approximately 26,000 tracks across 10 music genres. Each track is represented by features including:
- Acousticness
- Danceability
- Energy
- Instrumentalness
- Liveness
- Loudness
- Speechiness
- Tempo
- Valence

The dataset is well-balanced with each genre representing approximately 10% of the samples.

## Installation
To run this project, you'll need Python 3.6+ and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn category_encoders jupyter
```

## Project Structure
The project is organized into the following main sections:

1. **Data Audit**: Thorough examination of the dataset including shape, missing values, duplicates, and class distribution.
2. **Data Wrangling**: Preprocessing steps including:
   - Dropping non-predictive columns
   - Encoding categorical variables using target encoding
   - Scaling numerical features
3. **Model Building**: Training and optimizing two models:
   - Support Vector Machine (SVM)
   - Random Forest Classifier
4. **Evaluation**: Comprehensive evaluation using:
   - Cross-validation
   - Test set performance metrics
   - Classification reports
   - Confusion matrices

## Methodology

### Data Preprocessing
- Examined dataset for missing values and duplicates (none found)
- Analyzed class distribution (balanced across 10 genres)
- Encoded artist names using target encoding
- Scaled numerical features using RobustScaler
- Performed feature selection to identify most predictive attributes

### Model Training
The project implements two classification models:
1. **Support Vector Machine (SVM)**:
   - Used RBF kernel
   - Hyperparameters: C=5, gamma=0.05
   - Cross-validated with 5-fold CV

2. **Random Forest Classifier**:
   - Used 100 trees
   - Cross-validated with 5-fold CV
   - Identified most important features

## Results

### Model Performance
- **SVM**:
  - Cross-validation accuracy: 68.26% (±1.05%)
  - Test accuracy: 67.51%
  
- **Random Forest**:
  - Cross-validation accuracy: 71.28% (±1.53%)
  - Test accuracy: 71.03%

Random Forest outperformed SVM across all metrics, suggesting its ability to better capture the complex relationships between audio features and music genres.

### Feature Importance
The most significant audio features for genre classification were:
- Artist name (encoded)
- Popularity
- Acousticness
- Danceability
- Energy
- Instrumentalness
- Loudness
- Speechiness

## Future Improvements
Potential enhancements to this project could include:
- Implementing deep learning models (CNNs, RNNs)
- Incorporating additional audio features or raw audio analysis
- Exploring more advanced feature engineering techniques
- Implementing ensemble methods combining multiple models
- Adding a web interface for real-time genre prediction

## Technologies Used
- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Category Encoders**: Advanced categorical encoding
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment



## K-means Clustering Analysis on Red Wine Quality Dataset

In addition to the music genre classification, this project includes a K-means clustering analysis on the Red Wine Quality dataset to demonstrate clustering techniques.

### Dataset Information

The Red Wine Quality dataset contains physicochemical properties of red wine samples and their quality ratings. The dataset includes:
- 1,599 samples of red wine
- 11 chemical properties as features (such as acidity, sugar, pH, alcohol content)
- Quality ratings on a scale of 0-10

### Analysis Process

The K-means clustering implementation follows these steps:

1. **Data Preprocessing**
   - Loading the red wine quality dataset
   - Standardizing features to ensure equal contribution to clustering
   - Exploring data distribution and correlations

2. **Optimal Cluster Determination**
   - Using the Elbow Method to identify the optimal number of clusters
   - Calculating Silhouette Scores to validate clustering quality
   - Visualizing results to determine the best k value

3. **Cluster Analysis**
   - Applying K-means with the optimal number of clusters
   - Reducing dimensionality with PCA for visualization
   - Plotting clusters and comparing with original quality ratings
   - Analyzing feature importance for each cluster

### Key Findings

- Clusters reveal natural groupings in wine samples based on chemical properties
- Visualization shows how these clusters relate to the original quality ratings
- Analysis identifies which chemical properties most strongly influence cluster formation
- Correlation patterns within clusters provide insights into quality determinants

### Visualizations

- PCA plots showing cluster distributions
- Feature importance charts for each cluster
- Quality distribution across clusters
- Correlation heatmaps for each cluster

This additional analysis demonstrates how unsupervised learning techniques can be used to discover hidden patterns in datasets without predefined classifications.
---

This project was developed as part of a machine learning portfolio to demonstrate skills in data analysis, feature engineering, and classification algorithms.
