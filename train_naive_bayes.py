# train_naive_bayes.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load dataset
df = pd.read_csv("balanced_email_dataset.csv")

# 2. Ubah label dari teks ke angka
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Gunakan seluruh data dan acak
subset_df = df.sample(frac=1, random_state=42)

# 4. TF-IDF vektorisasi
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(subset_df['body'].astype(str))
y = subset_df['label']

# 5. Split: 10% test, sisanya (90%) untuk train+val
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# 6. Split dari 90%: 22.22% val (jadi totalnya 20%) → sisanya 70% train
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, random_state=42)

# 7. Latih model dengan data training
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Validasi model (opsional, misal untuk tuning)
y_val_pred = model.predict(X_val)
print("=== VALIDATION SET EVALUATION ===")
print(classification_report(y_val, y_val_pred, target_names=["Ham", "Spam"]))

# 9. Evaluasi pada test set
y_test_pred = model.predict(X_test)
print("\n=== FINAL TEST SET EVALUATION ===")
print(classification_report(y_test, y_test_pred, target_names=["Ham", "Spam"]))

# 10. Confusion Matrix (Test Set)
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()

# 11. ROC Curve dan AUC Score (Test Set)
y_test_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_probs)
roc_auc = roc_auc_score(y_test, y_test_probs)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 12. Simpan model dan vectorizer
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model dan vectorizer berhasil disimpan.")
