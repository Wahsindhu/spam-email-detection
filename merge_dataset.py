import os
import shutil

# Folder sumber dan tujuan
sources = {
    "enron_spam": "dataset/enron/spam",
    "enron_ham": "dataset/enron/ham",
    "spamassassin_spam": "dataset/spamassassin/spam",
    "spamassassin_ham": "dataset/spamassassin/ham",
}

target_base = "merged_dataset"
target_spam = os.path.join(target_base, "spam")
target_ham = os.path.join(target_base, "ham")

# Buat folder target jika belum ada
os.makedirs(target_spam, exist_ok=True)
os.makedirs(target_ham, exist_ok=True)

# Fungsi salin dan ubah nama file
def copy_and_rename(src_folder, dst_folder, prefix):
    files = [f for f in os.listdir(src_folder) if f.endswith(".txt")]
    for idx, filename in enumerate(files, start=1):
        src_path = os.path.join(src_folder, filename)
        dst_filename = f"{prefix}_{idx:04d}.txt"
        dst_path = os.path.join(dst_folder, dst_filename)
        shutil.copyfile(src_path, dst_path)

# Proses semua folder sumber
copy_and_rename(sources["enron_spam"], target_spam, "enron_spam")
copy_and_rename(sources["enron_ham"], target_ham, "enron_ham")
copy_and_rename(sources["spamassassin_spam"], target_spam, "spamassassin_spam")
copy_and_rename(sources["spamassassin_ham"], target_ham, "spamassassin_ham")

print("✅ Penggabungan dataset selesai.")
