# Titanic Survival Prediction

A machine learning project that predicts passenger survival in the Titanic disaster using the classic Titanic dataset.

---

## Project Overview

This project builds a **Random Forest Classifier** to predict whether a passenger survived the Titanic disaster based on features like passenger class, gender, age, fare, and family size.

---

## Dataset

**Source:** `Titanic-Dataset.csv`  
**Size:** 891 passengers × 12 features

### Features

| Feature | Description |
|---|---|
| `PassengerId` | Unique identifier |
| `Survived` | Target variable (0 = No, 1 = Yes) |
| `Pclass` | Passenger class (1, 2, 3) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C / Q / S) |

---

## Project Workflow

### 1. Data Loading & Exploration
- Loaded dataset and inspected shape, dtypes, and summary statistics
- Identified missing values: **Age** (177), **Cabin** (687), **Embarked** (2)

### 2. Data Preprocessing
- Imputed missing **Age** and **Fare** values with column means
- Confirmed no duplicate records
- Dropped non-informative columns: `PassengerId`, `Name`, `Ticket`
- Encoded categorical variables:
  - `Sex`: male → 1, female → 0
  - `Embarked`: Q → 0, S → 1, C → 2

### 3. Exploratory Data Analysis

Key findings:

- **Survival rate:** ~38% survived, ~62% did not
- **Gender:** Female survival rate (~74%) far exceeded male (~19%)
- **Passenger Class:** First-class passengers had significantly higher survival rates
- **Age:** Younger passengers showed slightly better survival odds
- **Family size:** Solo travelers had lower survival rates

### 4. Model Training

- **Algorithm:** Random Forest Classifier
- **Split:** Train/test split
- **Preprocessing:** Label encoding, standard scaling, mean imputation

---

## Libraries Used

```python
pandas
numpy
seaborn
matplotlib
scikit-learn
```

---

## How to Run

1. Clone this repository

```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Install dependencies

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

3. Add the dataset

Place `Titanic-Dataset.csv` in the project root directory.

4. Run the notebook

Open `Titanic_Survival_codsoft.ipynb` in Jupyter Notebook or Google Colab and run all cells.

---

## Key Findings

| Factor | Insight |
|---|---|
| Overall survival | 38.38% survived |
| Gender | Female survival ~74%, Male ~19% |
| Passenger class | 1st class had highest survival rate |
| Traveling alone | Lower chance of survival |

---

## Results

- The Random Forest model was trained on the preprocessed Titanic dataset
- Evaluated using accuracy score, classification report, and confusion matrix

---

## Project Structure
