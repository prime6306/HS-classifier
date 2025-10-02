# 📦 HS Code Classification with Machine Learning

This project automates the classification of product descriptions into HS (Harmonized System) Codes using natural language processing and machine learning. It is designed to support trade, customs, and logistics operations by reducing manual errors and improving classification speed.

---

## 🔍 Problem Statement

Manual HS code classification is time-consuming and error-prone. By training machine learning models on product descriptions, we aim to predict the correct HS code automatically, focusing on the **top 20 most frequent headings** for better model balance and relevance.

---

## 🧰 Tools & Libraries

- Python 3.10+
- `pandas`, `numpy`
- `scikit-learn`
- `nltk`
- `matplotlib`, `seaborn`
- `pickle`

---

## 🧪 Workflow

1. **Data Preprocessing**
   - Filled missing headings using forward fill.
   - Filtered to top 20 most frequent HS headings.
   - Cleaned and normalized text.

2. **Feature Engineering**
   - Used **TF-IDF Vectorization** to convert text into numerical features.

3. **Model Training & Tuning**
   - Trained and compared:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
     - Gradient Boosting
   - Used **GridSearchCV** for hyperparameter tuning.

4. **Evaluation**
   - Accuracy, Precision, Recall
   - Confusion Matrix & Classification Report

5. **Exporting Models**
   - Saved the final **SVM** and **Random Forest** models with the TF-IDF vectorizer using `pickle`.

---

## 🎯 Model Performance

| Model            | Accuracy | Precision | Recall   |
|------------------|----------|-----------|----------|
| SVM              | 92.97%   | ~85.08%   | ~85.48%  |
| Random Forest    | 88.83%   | ~86.19%   | ~86.12%  |
| Gradient Boosting| 73.15%   | ~85.87%   | ~83.68%  |

> ✅ SVM was selected as the best model based on highest overall performance.

---

## 💾 Output Files

- `model_svm.pkl` – Trained SVM model  
- `model_rf.pkl` – Trained Random Forest model  
- `vectorizer.pkl` – Fitted TF-IDF Vectorizer  
- `confusion_matrix.png` – Model confusion matrix (SVM)

---
