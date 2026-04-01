import os
import shutil

# 🔧 CAMINHOS
src_images = r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\frames"
src_labels = r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\datasets\labels\train"

dst_images = r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\dataset\images\train"
dst_labels = r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\dataset\labels\train"


def get_unique_name(filename, existing_files):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_name = filename

    while new_name in existing_files:
        new_name = f"{name}_{counter}{ext}"
        counter += 1

    return new_name


def main():
    src_files = os.listdir(src_images)
    dst_files = set(os.listdir(dst_images))

    copied = 0
    skipped = 0

    for file in src_files:
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        src_img_path = os.path.join(src_images, file)
        label_name = os.path.splitext(file)[0] + ".txt"
        src_lbl_path = os.path.join(src_labels, label_name)

        # ⚠️ garantir que existe label correspondente
        if not os.path.exists(src_lbl_path):
            print(f"[SKIP] Sem label: {file}")
            skipped += 1
            continue

        # 🔁 gerar nome único
        new_img_name = get_unique_name(file, dst_files)
        new_lbl_name = os.path.splitext(new_img_name)[0] + ".txt"

        dst_img_path = os.path.join(dst_images, new_img_name)
        dst_lbl_path = os.path.join(dst_labels, new_lbl_name)

        # 📂 copiar ficheiros
        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_lbl_path, dst_lbl_path)

        dst_files.add(new_img_name)

        copied += 1
        print(f"[OK] {file} -> {new_img_name}")

    print("\n--- RESULTADO ---")
    print(f"Copiados: {copied}")
    print(f"Skipped (sem label): {skipped}")


if __name__ == "__main__":
    main()