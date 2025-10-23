# Breast Cancer Classification using Linear Discriminant Analysis (LDA)

A machine learning project focused on the classification of breast masses into benign or malignant categories using classic dimensionality reduction and classification algorithms.

---

## Problem Statement / Goal

The primary goal is to develop and evaluate a machine learning model capable of accurately classifying breast masses as **Benign (0)** or **Malignant (1)** using key cellular and nuclear features from the Wisconsin Breast Cancer (Diagnostic) dataset.

A secondary goal is to explore the effectiveness of **Linear Discriminant Analysis (LDA)** as both a feature reduction technique and a classifier for this two-class problem.

---

## Tech Stack / Tools Used

The project is implemented in Python and relies on the following core libraries:

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python | Primary programming language |
| **Data Handling** | Pandas, NumPy | Data structure and numerical operations |
| **Modeling** | Scikit-learn | Machine learning model implementation (LDA, SGDClassifier, etc.) |
| **Preprocessing**| StandardScaler, train_test_split | Feature scaling and data splitting |

---

## Approach / Methodology

1.  **Data Loading and Inspection**: Loaded the Wisconsin Breast Cancer dataset from `sklearn.datasets`.
2.  **Feature Transformation**: Applied a log-transformation (`np.log1p`) to skewed features for distribution normalization.
3.  **Feature Scaling**: Standardized all features using **StandardScaler**.
4.  **Data Splitting**: Split the data into training and testing sets (80% train, 20% test).
5.  **Model Implementation**: Trained **Linear Discriminant Analysis (LDA)** as the primary model and **Stochastic Gradient Descent (SGD) Classifier** for comparison.
6.  **Evaluation**: Assessed performance using Accuracy, Precision, Recall, and F1-Score.

---

## Results / Key Findings

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **LDA** | 97.37% | 95.95% | 100.0% | 98.59% |
| **SGD Classifier** | **98.25%** | **98.59%** | **98.59%** | **98.59%** |

* The **SGD Classifier** achieved the highest accuracy at **~98.25%**.
* The **LDA model** achieved a perfect **Recall of 1.0**, successfully identifying all Malignant cases in the test set.

---

## Topic Tags

`Machine Learning`, `Classification`, `Linear Discriminant Analysis (LDA)`, `SGD Classifier`, `Breast Cancer`, `Data Science`, `Python`, `Scikit-learn`, `Data Preprocessing`.

---

## How to Run the Project

### 1. Install Requirements

Install all necessary packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
