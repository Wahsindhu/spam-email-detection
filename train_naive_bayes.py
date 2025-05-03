# train_naive_bayes.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Untuk simpan dan load model

# 1. Load dataset
df = pd.read_csv("balanced_email_dataset.csv")

# 2. Ubah label dari teks ke angka
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Ambil 500 spam dan 500 ham
spam_df = df[df['label'] == 1].sample(500, random_state=42)
ham_df = df[df['label'] == 0].sample(500, random_state=42)
subset_df = pd.concat([spam_df, ham_df]).sample(frac=1, random_state=42)  # acak ulang

# 4. Ubah teks menjadi fitur TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(subset_df['body'].astype(str))
y = subset_df['label']

# 5. Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Inisialisasi dan latih model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Prediksi dan evaluasi
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 8. Visualisasi confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 9. Simpan model dan vectorizer
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model dan vectorizer berhasil disimpan.")
