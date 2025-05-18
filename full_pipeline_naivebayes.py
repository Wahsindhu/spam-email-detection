import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load dataset
df = pd.read_csv("combined_email_dataset.csv")

# 2. Gabungkan kolom subject + body jadi satu teks
df["text"] = df["subject"].fillna('') + " " + df["body"].fillna('')

# 3. TF-IDF vektorisasi
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["text"])

# 4. Metadata TF-IDF
print("TF-IDF shape:", tfidf_matrix.shape)
print("Jumlah fitur unik:", len(vectorizer.get_feature_names_out()))
print("Contoh fitur:", vectorizer.get_feature_names_out()[:20])

# 5. Siapkan label
y = df["label"]

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)

# 7. Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Evaluasi
y_pred = model.predict(X_test)
print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 10. (Opsional) Simpan model dan vectorizer
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model dan vectorizer berhasil disimpan.")
