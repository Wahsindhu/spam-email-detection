import pandas as pd
import matplotlib.pyplot as plt

# Baca dataset
csv_path = "combined_email_dataset.csv"
df = pd.read_csv(csv_path)

# Hitung distribusi label
label_counts = df['label'].value_counts()

# Tampilkan jumlah
print("Distribusi Dataset:")
for label, count in label_counts.items():
    print(f"{label.capitalize()}: {count} email")

# Visualisasi
labels = label_counts.index.tolist()
counts = label_counts.values.tolist()

plt.bar(labels, counts, color=['red', 'green'])
plt.title("Distribusi Email Spam vs Ham")
plt.ylabel("Jumlah Email")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
