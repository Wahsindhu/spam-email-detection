# 📧 Spam Email Detection using Machine Learning

Proyek ini merupakan implementasi deteksi email spam menggunakan metode machine learning berbasis Python. Dataset yang digunakan merupakan gabungan dari dua dataset open-source terkenal, yaitu **SpamAssassin** dan **Enron Email Dataset**.
Proyek ini dibuat untuk pembelajaran dan eksperimen dalam mendeteksi email spam menggunakan teknik machine learning dasar.
## 📂 Struktur Dataset

Dataset mentah terdiri dari file `.eml` atau `.txt` yang dikategorikan ke dalam folder:


## ⚙️ Fitur dan Alur Proyek

### 1. Gabung Dataset (`combine_email_dataset.py`)

- Mengekstrak metadata dan isi email dari kedua dataset (SpamAssassin dan Enron).
- Menggabungkan semuanya menjadi satu file CSV: `combined_email_dataset.csv`.

### 2. Balancing dan Ekstraksi Fitur (`balance_dataset.py`)

- Menyeimbangkan jumlah data **spam** dan **ham** agar model tidak bias.
- Melakukan ekstraksi fitur menggunakan **TF-IDF Vectorizer**.
- Menyimpan hasil sebagai:
  - `balanced_email_dataset.csv` → dataset seimbang
  - `labels.csv` → label target (spam/ham)
  - `tfidf_features.pkl` → fitur numerik hasil TF-IDF

## 📦 Output Files

| File | Deskripsi |
|------|-----------|
| `combined_email_dataset.csv` | Dataset gabungan dari Enron dan SpamAssassin |
| `balanced_email_dataset.csv` | Dataset seimbang (jumlah spam = ham) |
| `tfidf_features.pkl` | Matriks TF-IDF hasil ekstraksi fitur |
| `labels.csv` | Label klasifikasi untuk training model |

## 📚 Dependencies

Proyek ini dibangun menggunakan Python 3 dan library berikut:

- `pandas`
- `sklearn`
- `email` (builtin Python)
- `pickle`
- `os`

Instalasi environment (opsional):
```bash
pip install -r requirements.txt
