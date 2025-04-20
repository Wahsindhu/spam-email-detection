import os

spamassassin_dir = 'dataset/spamassassin'

for root, dirs, files in os.walk(spamassassin_dir):
    for file in files:
        old_path = os.path.join(root, file)
        # Tambahkan .txt kalau belum ada
        if not file.endswith('.txt'):
            new_path = os.path.join(root, file + '.txt')
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} -> {new_path}')
