import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
import pickle

# Load data
df = pd.read_csv("combined_email_dataset.csv")

# Bersihkan data
df = df.dropna(subset=["body"])
df["source"] = df["source"].str.lower()
df["label"] = df["label"].str.lower()

print("📚 Sumber data unik:", df["source"].unique())

# Pisahkan berdasarkan sumber dan label
df_spamassassin = df[df["source"] == "spamassassin"]
df_enron = df[df["source"] == "enron"]

# Hitung jumlah yang tersedia dari SpamAssassin
spam_sa = df_spamassassin[df_spamassassin["label"] == "spam"]
ham_sa = df_spamassassin[df_spamassassin["label"] == "ham"]

print("Jumlah SpamAssassin spam:", len(spam_sa))
print("Jumlah SpamAssassin ham :", len(ham_sa))

# Target total
target_per_class = 10000

# Hitung kekurangan dari Enron
def get_extra(df_main, df_backup, label, total_target):
    n_main = len(df_main)
    needed = total_target - n_main
    if needed <= 0:
        return resample(df_main, replace=False, n_samples=total_target, random_state=42)
    else:
        df_extra = df_backup[df_backup["label"] == label]
        df_extra_sample = resample(df_extra, replace=False, n_samples=needed, random_state=42)
        return pd.concat([df_main, df_extra_sample])

# Ambil spam dan ham sesuai target
spam_final = get_extra(spam_sa, df_enron, "spam", target_per_class)
ham_final = get_extra(ham_sa, df_enron, "ham", target_per_class)

# Gabung dan acak
df_balancing = pd.concat([spam_final, ham_final]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\n📊 Komposisi akhir (Spam + Ham):")
print(df_balancing["label"].value_counts())

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df_balancing["body"])

print(f"\n✅ TF-IDF berhasil dibuat. Shape: {X_tfidf.shape}")

# Simpan
df_balancing.to_csv("balanced_email_dataset.csv", index=False)
with open("tfidf_features.pkl", "wb") as f:
    pickle.dump(X_tfidf, f)

df_balancing["label"].to_csv("labels.csv", index=False)

print("📁 Dataset seimbang disimpan sebagai balanced_email_dataset.csv")
print("📦 TF-IDF matrix disimpan sebagai tfidf_features.pkl")
print("📑 Label disimpan sebagai labels.csv")
