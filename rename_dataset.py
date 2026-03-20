import os

IMAGES_DIR = r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\dataset\images\train"
LABELS_DIR = r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\dataset\labels\train"

files = sorted(os.listdir(IMAGES_DIR))

# 🔥 FASE 1 — nomes temporários (evita conflitos)
temp_names = []

for i, file in enumerate(files):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    old_img_path = os.path.join(IMAGES_DIR, file)
    old_label_path = os.path.join(LABELS_DIR, os.path.splitext(file)[0] + ".txt")

    temp_img_name = f"temp_{i}.jpg"
    temp_label_name = f"temp_{i}.txt"

    temp_img_path = os.path.join(IMAGES_DIR, temp_img_name)
    temp_label_path = os.path.join(LABELS_DIR, temp_label_name)

    os.rename(old_img_path, temp_img_path)

    if os.path.exists(old_label_path):
        os.rename(old_label_path, temp_label_path)

    temp_names.append((temp_img_name, temp_label_name))

# 🔥 FASE 2 — nomes finais
for i, (temp_img_name, temp_label_name) in enumerate(temp_names):

    temp_img_path = os.path.join(IMAGES_DIR, temp_img_name)
    temp_label_path = os.path.join(LABELS_DIR, temp_label_name)

    final_img_name = f"frame_{i}.jpg"
    final_label_name = f"frame_{i}.txt"

    final_img_path = os.path.join(IMAGES_DIR, final_img_name)
    final_label_path = os.path.join(LABELS_DIR, final_label_name)

    os.rename(temp_img_path, final_img_path)

    if os.path.exists(temp_label_path):
        os.rename(temp_label_path, final_label_path)

print(f"Renomeados {len(temp_names)} ficheiros com sucesso!")