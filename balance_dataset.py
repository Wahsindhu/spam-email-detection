import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
import joblib

# 1. Baca dataset gabungan
df = pd.read_csv("combined_email_dataset.csv")

# 2. Cek jumlah awal spam dan ham
spam = df[df['label'] == 'spam']
ham = df[df['label'] == 'ham']
print("Jumlah spam:", len(spam))
print("Jumlah ham:", len(ham))

# 3. Samakan jumlah data spam dan ham (undersampling)
min_len = min(len(spam), len(ham))
spam_downsampled = resample(spam, replace=False, n_samples=min_len, random_state=42)
ham_downsampled = resample(ham, replace=False, n_samples=min_len, random_state=42)

# 4. Gabungkan kembali data yang telah diseimbangkan
df_balancing = pd.concat([spam_downsampled, ham_downsampled])
df_balancing = df_balancing.sample(frac=1, random_state=42).reset_index(drop=True)  # acak baris

print("\nSetelah balancing:")
print(df_balancing["label"].value_counts())

# 5. Hapus nilai kosong di kolom 'body'
df_balancing['body'] = df_balancing['body'].fillna('')

# 6. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df_balancing["body"])

print(f"\n✅ TF-IDF berhasil dibuat. Shape: {X_tfidf.shape}")

# 7. Simpan dataset hasil balancing ke file
df_balancing.to_csv("balanced_email_dataset.csv", index=False)
print("📁 Dataset seimbang disimpan sebagai balanced_email_dataset.csv")

# 8. Simpan TF-IDF dan label untuk keperluan training nanti
joblib.dump(X_tfidf, "tfidf_features.pkl")
df_balancing['label'].to_csv("labels.csv", index=False)
print("📦 TF-IDF matrix disimpan sebagai tfidf_features.pkl")
print("📑 Label disimpan sebagai labels.csv")
