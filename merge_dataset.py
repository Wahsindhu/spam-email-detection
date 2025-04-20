import os
from email import policy
from email.parser import BytesParser
import pandas as pd

def extract_email_data(path, label, source):
    data = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.eml') or file.endswith('.txt'):
                try:
                    with open(os.path.join(root, file), 'rb') as f:
                        msg = BytesParser(policy=policy.default).parse(f)
                        body = ''
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == 'text/plain':
                                    body = part.get_payload(decode=True).decode(errors='ignore')
                                    break
                        else:
                            body = msg.get_payload(decode=True)
                            if body:
                                body = body.decode(errors='ignore')

                        data.append({
                            'from': msg['From'],
                            'to': msg['To'],
                            'subject': msg['Subject'],
                            'date': msg['Date'],
                            'body': body.strip() if body else '',
                            'label': label,
                            'source': source
                        })
                except Exception as e:
                    print(f"Gagal membaca {file}: {e}")
    return data

# Ganti path sesuai struktur folder kamu
enron_spam_path = 'dataset/enron/spam'
enron_ham_path = 'dataset/enron/ham'

spamassassin_spam_path = 'dataset/spamassassin/spam'
spamassassin_ham_path = 'dataset/spamassassin/ham'

print("⏳ Memproses Enron...")
enron_spam = extract_email_data(enron_spam_path, 'spam', 'enron')
enron_ham = extract_email_data(enron_ham_path, 'ham', 'enron')

print("⏳ Memproses SpamAssassin...")
spamassassin_spam = extract_email_data(spamassassin_spam_path, 'spam', 'spamassassin')
spamassassin_ham = extract_email_data(spamassassin_ham_path, 'ham', 'spamassassin')

print("✅ Menggabungkan dataset...")
combined_data = enron_spam + enron_ham + spamassassin_spam + spamassassin_ham
df = pd.DataFrame(combined_data)

print(df['source'].value_counts())  # <-- Tambahan untuk verifikasi

output_file = 'combined_email_dataset.csv'
df.to_csv(output_file, index=False)
print(f"✅ Dataset gabungan berhasil disimpan sebagai {output_file}")
