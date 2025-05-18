import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset (pastikan file .csv ada di folder yang sama atau ubah path-nya)
df = pd.read_csv("combined_email_dataset.csv")

# Gabungkan subject dan body jadi satu kolom teks
df["text"] = df["subject"].fillna('') + " " + df["body"].fillna('')

# TF-IDF vektorisasi
vectorizer = TfidfVectorizer(max_features=5000)  # Boleh disesuaikan jumlah fiturnya
tfidf_matrix = vectorizer.fit_transform(df["text"])

# Tampilkan metadata
print("TF-IDF shape:", tfidf_matrix.shape)  # baris = jumlah dokumen, kolom = fitur
print("Jumlah fitur unik:", len(vectorizer.get_feature_names_out()))
print("Contoh fitur:", vectorizer.get_feature_names_out()[:20])
