# 📧 Spam Email Detection — Machine Learning Project

## 📌 Overview
This project builds a machine learning model to automatically classify messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques.

The model is trained using TF-IDF vectorization and a Multinomial Naive Bayes classifier, with hyperparameter tuning and cross-validation to ensure strong generalization performance.

---

## 🎯 Objective
To develop a robust spam detection system that:
- Accurately detects spam messages
- Minimizes false positives
- Generalizes well to unseen data
- Can be reused for future predictions

---

## 📂 Dataset
The dataset contains labeled SMS messages:
- `ham (0)` → legitimate messages
- `spam (1)` → unwanted/promotional messages

Class distribution:
- Ham: 4825
- Spam: 747

---

## 🛠 Project Pipeline

### 1️⃣ Data Preprocessing
- Lowercasing text
- Removing punctuation
- Removing stopwords
- Stemming
- Feature engineering:
  - Message length
  - Number of digits
  - Number of exclamation marks

### 2️⃣ Feature Extraction
- TF-IDF Vectorization (unigrams + bigrams)
- Additional numerical features converted to categorical bins

### 3️⃣ Model
- Multinomial Naive Bayes
- Hyperparameter tuning using GridSearchCV

### 4️⃣ Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- 5-Fold Cross-Validation

---

## 📊 Model Performance


**Test Set Results:**
- Accuracy: 0.99
- Spam Precision: 0.97
- Spam Recall: 0.92
- Spam F1-score: 0.94

**Cross-Validation Results (5-Fold CV on Training Set):**
- F1 scores per fold: [0.906, 0.950, 0.918, 0.932, 0.956]
- Mean F1-score: 0.933

> The close performance between Cross-Validation and Test scores indicates that the model does **not overfit** and generalizes well to unseen data.
