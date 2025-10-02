
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1. Load dataset
df = pd.read_excel("dataset4.xlsx")

# Fill missing Heading with forward fill
df['Heading'] = df['Heading'].fillna(method='ffill')
df.dropna(subset=['Heading', 'Description'], inplace=True)

# Optional: Limit to top 20 classes
top_classes = df['Heading'].value_counts().nlargest(20).index
df = df[df['Heading'].isin(top_classes)]

# 2. Split dataset
X = df['Description']
y = df['Heading']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train Best SVM (C=1, kernel='linear')
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_vec, y_train)

# 5. Train Best Random Forest (n_estimators=200, max_depth=None)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf_model.fit(X_train_vec, y_train)

# 6. Save models and vectorizer
with open("best_svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("best_rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Models and vectorizer saved!")

# 7. Predict and Evaluate - SVM
y_pred_svm = svm_model.predict(X_test_vec)
print("\n SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Purples", xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 8. Predict and Evaluate - Random Forest
y_pred_rf = rf_model.predict(X_test_vec)
print("\n Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf_model.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
