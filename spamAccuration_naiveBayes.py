import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Membaca dataset
df = pd.read_csv("combined_email_dataset.csv")

# 2. Mengecek kolom yang tersedia
print(df.columns)

# 3. Asumsi kolom teks email adalah 'text' dan label adalah 'label' (ubah jika berbeda)
# Pastikan kolom label-nya bernilai 0 (ham) dan 1 (spam), ubah jika perlu
text_col = 'text'   # ganti sesuai nama kolom di file
label_col = 'label' # ganti sesuai nama kolom di file

# 4. Preprocessing dan vectorisasi
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df[text_col])
y = df[label_col]

# 5. Split data training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Training dengan Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nKlasifikasi:\n", classification_report(y_test, y_pred))
