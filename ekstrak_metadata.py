import pandas as pd
import numpy as np
from email.utils import parsedate_to_datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tldextract
import re
import joblib

# 1. Baca dataset
df = pd.read_csv("balanced_email_dataset.csv")

# 2. Ekstrak alamat email dari kolom 'from'
def extract_email_domain(from_field):
    if pd.isnull(from_field):
        return "unknown"
    match = re.search(r'<([^>]+)>', from_field)
    email_address = match.group(1) if match else from_field
    ext = tldextract.extract(email_address)
    return ext.domain if ext.domain else "unknown"

df['sender_domain'] = df['from'].apply(extract_email_domain)

# 3. Ekstrak jam dan hari dari kolom 'date'
def extract_hour(date_str):
    try:
        return parsedate_to_datetime(date_str).hour
    except Exception:
        return -1

def extract_day_of_week(date_str):
    try:
        return parsedate_to_datetime(date_str).weekday()
    except Exception:
        return -1

df['hour'] = df['date'].apply(extract_hour)
df['day_of_week'] = df['date'].apply(extract_day_of_week)

# 4. Panjang subject
df['subject_length'] = df['subject'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

# 5. One-hot encoding untuk domain pengirim
top_domains = df['sender_domain'].value_counts().nlargest(10).index
df['sender_domain'] = df['sender_domain'].apply(lambda x: x if x in top_domains else 'other')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
domain_encoded = encoder.fit_transform(df[['sender_domain']])

# 6. Fitur numerik lainnya
numerical_features = df[['hour', 'day_of_week', 'subject_length']]
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_features)

# 7. Gabungkan semua metadata
metadata_features = np.hstack([domain_encoded, numerical_scaled])

# 8. Simpan ke metadata_features.csv
metadata_df = pd.DataFrame(metadata_features)
metadata_df.to_csv("metadata_features.csv", index=False)

# Simpan encoder dan scaler
joblib.dump(encoder, "domain_encoder.pkl")
joblib.dump(scaler, "metadata_scaler.pkl")

print("✅ Metadata berhasil diekstrak dan disimpan ke metadata_features.csv")