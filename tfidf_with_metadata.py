import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
import re

# 1. Baca dataset
df = pd.read_csv("combined_email_dataset.csv")

# 2. Gabungkan subject dan body jadi satu teks
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# 3. Ekstrak metadata
def extract_metadata(text):
    text = str(text)
    return pd.Series({
        "body_length": len(text),
        "num_exclamation": text.count("!"),
        "num_links": len(re.findall(r"http[s]?://", text)),
        "num_uppercase": sum(1 for c in text if c.isupper())
    })

metadata_features = df["text"].apply(extract_metadata)
print("Contoh metadata:\n", metadata_features.head())

# 4. TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["text"])

# 5. Gabungkan TF-IDF dan metadata
X_combined = hstack([tfidf_matrix, metadata_features.values])

# 6. Label (pastikan kolom label-nya bernama 'label')
y = df["label"]

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 8. Latih model
model = MultinomialNB()
model.fit(X_train, y_train)

# 9. Evaluasi
y_pred = model.predict(X_test)

print("\n=== Hasil Evaluasi Model ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))
