# Movie Rating Prediction Using Machine Learning

Predicts IMDb movie ratings using regression on a dataset of Indian films
with features like genre, director, and cast.

## Overview

Built and evaluated a Linear Regression model on 5,659+ Indian movies
sourced from IMDb. The pipeline covers full data preprocessing, feature
engineering, and exploratory analysis.

## Dataset

- **Source:** IMDb Indian Movies dataset (imdb.csv)
- **Size:** 5,659 records after cleaning
- **Features:** Name, Year, Duration, Genre, Rating, Votes, Director, Actor 1,
  Actor 2, Actor 3

## Key Steps

- Removed duplicates and null values
- Parsed Duration (e.g., "109 min" → 109) and Votes (e.g., "1,086" → 1086)
  into numeric format
- Split Genre into Genre1, Genre2, Genre3 for multi-genre films
- Performed EDA: top directors, top actors, genre distribution, histograms,
  scatter plots, and Spearman correlation matrix

## Key Findings

- Drama is the most frequent genre in Indian cinema
- A small set of directors and actors dominate high-volume filmmaking
- Rating distribution centers around 5.9 (mean), ranging from 1.1 to 10.0

## Tech Stack

- Python, Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn, Plotly
- Scikit-learn (LinearRegression, train_test_split, MAE, MSE, R²)

## Project Structure
