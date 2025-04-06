import os
import matplotlib.pyplot as plt

# Lokasi folder hasil gabungan
spam_dir = "merged_dataset/spam"
ham_dir = "merged_dataset/ham"

# Hitung jumlah file .txt di setiap folder
spam_count = len([f for f in os.listdir(spam_dir) if f.endswith(".txt")])
ham_count = len([f for f in os.listdir(ham_dir) if f.endswith(".txt")])

# Tampilkan jumlah
print("Distribusi Dataset:")
print(f"📧 Spam: {spam_count} email")
print(f"📨 Ham: {ham_count} email")

# Visualisasi
labels = ['Spam', 'Ham']
counts = [spam_count, ham_count]

plt.bar(labels, counts, color=['red', 'green'])
plt.title("Distribusi Email Spam vs Ham")
plt.ylabel("Jumlah Email")
plt.grid(axis='y')
plt.tight_layout()
plt.show()