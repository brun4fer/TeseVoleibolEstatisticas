import os
import re
import shutil
from pathlib import Path

# =========================
# CONFIGURAÇÃO
# =========================
BASE_DIR = Path(r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\dataset")

TRAIN_IMAGES = BASE_DIR / "images" / "train"
TRAIN_LABELS = BASE_DIR / "labels" / "train"
VAL_IMAGES = BASE_DIR / "images" / "val"
VAL_LABELS = BASE_DIR / "labels" / "val"

VAL_RATIO = 0.20
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# True = move para val
# False = copia para val e mantém também em train
MOVE_FILES = True


def natural_key(name: str):
    """
    Ordenação natural:
    frame_2 antes de frame_10
    """
    parts = re.split(r"(\d+)", name.lower())
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def extract_base_stem(path: Path) -> str:
    return path.stem


def find_image_label_pairs(images_dir: Path, labels_dir: Path):
    pairs = []

    for img_path in images_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = extract_base_stem(img_path)
        label_path = labels_dir / f"{stem}.txt"

        if not label_path.exists():
            print(f"[SKIP] Sem label correspondente: {img_path.name}")
            continue

        pairs.append((img_path, label_path))

    return pairs


def ensure_dirs():
    VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    VAL_LABELS.mkdir(parents=True, exist_ok=True)


def destination_exists(img_name: str, lbl_name: str) -> bool:
    return (VAL_IMAGES / img_name).exists() or (VAL_LABELS / lbl_name).exists()


def move_or_copy(src: Path, dst: Path, move_files: bool):
    if move_files:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    ensure_dirs()

    pairs = find_image_label_pairs(TRAIN_IMAGES, TRAIN_LABELS)

    if not pairs:
        print("Não foram encontrados pares imagem+label em train.")
        return

    # Ordenar por nome natural para manter blocos temporais
    pairs.sort(key=lambda pair: natural_key(pair[0].stem))

    total = len(pairs)
    val_count = max(1, int(total * VAL_RATIO))
    split_index = total - val_count

    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    print(f"Total de pares encontrados: {total}")
    print(f"Train ficará com: {len(train_pairs)}")
    print(f"Val receberá: {len(val_pairs)}")
    print(f"Modo: {'MOVE' if MOVE_FILES else 'COPY'}")
    print("-" * 50)

    moved = 0
    skipped_existing = 0

    for img_path, lbl_path in val_pairs:
        dst_img = VAL_IMAGES / img_path.name
        dst_lbl = VAL_LABELS / lbl_path.name

        if destination_exists(img_path.name, lbl_path.name):
            print(f"[SKIP] Já existe em val: {img_path.name}")
            skipped_existing += 1
            continue

        move_or_copy(img_path, dst_img, MOVE_FILES)
        move_or_copy(lbl_path, dst_lbl, MOVE_FILES)

        moved += 1
        print(f"[OK] {img_path.name} -> val")

    print("\nResultado final")
    print(f"Pares enviados para val: {moved}")
    print(f"Saltados por já existirem: {skipped_existing}")


if __name__ == "__main__":
    main()