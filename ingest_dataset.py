"""
ingest_dataset.py
-----------------
Importa novos pares (imagem, label YOLO) para o dataset estruturado.

Origem:
  - Imagens:  C:\\Users\\Utilizador\\Documents\\GitHub\\TeseVoleibolEstatisticas\\frames
  - Labels:   C:\\Users\\Utilizador\\Desktop\\Mestrado\\Tese\\obj_train_data
  (cada label .txt deve corresponder a uma imagem com o mesmo stem)

Destino:
  C:\\Users\\Utilizador\\Documents\\GitHub\\TeseVoleibolEstatisticas\\dataset
    images/train , images/val
    labels/train , labels/val

Colisões:
  Se já existir um ficheiro com o mesmo nome em qualquer dos 4 directórios
  do dataset (images/train, images/val, labels/train, labels/val), o novo
  par é renomeado adicionando o sufixo "_N" (frame_0.jpg → frame_0_1.jpg,
  frame_0_2.jpg, …). O sufixo é o mesmo para a imagem e para o label.

Modo:
  Por defeito copia (mantém os originais intactos). Para mover, mudar
  MOVE_FILES = True.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Tuple


# =========================
# CONFIGURAÇÃO
# =========================
SRC_IMAGES_DIR = Path(r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\frames")
SRC_LABELS_DIR = Path(r"C:\Users\Utilizador\Desktop\Mestrado\Tese\obj_train_data")

DATASET_DIR = Path(r"C:\Users\Utilizador\Documents\GitHub\TeseVoleibolEstatisticas\dataset")
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
TRAIN_LABELS = DATASET_DIR / "labels" / "train"
VAL_IMAGES = DATASET_DIR / "images" / "val"
VAL_LABELS = DATASET_DIR / "labels" / "val"

VAL_RATIO = 0.20
RANDOM_SEED = 42  # reprodutibilidade do split
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MOVE_FILES = False  # True = move (apaga o original); False = copia


def find_pairs() -> List[Tuple[Path, Path]]:
    """Empareia cada imagem com o label .txt de stem igual em SRC_LABELS_DIR."""
    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(SRC_IMAGES_DIR.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = SRC_LABELS_DIR / f"{img.stem}.txt"
        if not lbl.exists():
            print(f"[SKIP] Sem label: {img.name}")
            continue
        pairs.append((img, lbl))
    return pairs


def collect_taken_names() -> set:
    """Conjunto de stems já existentes em qualquer dos 4 directórios destino."""
    taken = set()
    for d in (TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS):
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    taken.add(f.stem)
    return taken


def resolve_unique_stem(stem: str, taken: set) -> str:
    """Devolve o primeiro stem livre da forma stem, stem_1, stem_2, …

    Mantém a convenção observada no dataset (`frame_100_1.jpg`).
    Atualiza `taken` com o stem escolhido para evitar colisões dentro deste run.
    """
    if stem not in taken:
        taken.add(stem)
        return stem
    n = 1
    while f"{stem}_{n}" in taken:
        n += 1
    new_stem = f"{stem}_{n}"
    taken.add(new_stem)
    return new_stem


def transfer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if MOVE_FILES:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main() -> None:
    if not SRC_IMAGES_DIR.is_dir():
        raise SystemExit(f"Pasta de imagens não existe: {SRC_IMAGES_DIR}")
    if not SRC_LABELS_DIR.is_dir():
        raise SystemExit(f"Pasta de labels não existe: {SRC_LABELS_DIR}")

    for d in (TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS):
        d.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs()
    if not pairs:
        print("Nenhum par imagem+label encontrado.")
        return

    # Split estável: shuffle com seed e fatia 80/20.
    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(pairs)
    val_count = max(1, int(round(len(pairs) * VAL_RATIO)))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    taken = collect_taken_names()

    print(f"Pares encontrados: {len(pairs)}")
    print(f"  → train: {len(train_pairs)}")
    print(f"  → val:   {len(val_pairs)}")
    print(f"Modo: {'MOVE' if MOVE_FILES else 'COPY'}")
    print(f"Stems já ocupados no destino: {len(taken)}")
    print("-" * 60)

    renamed = 0
    written = 0

    def ingest(batch: List[Tuple[Path, Path]], img_dst_dir: Path, lbl_dst_dir: Path, label: str) -> None:
        nonlocal renamed, written
        for img_src, lbl_src in batch:
            original_stem = img_src.stem
            new_stem = resolve_unique_stem(original_stem, taken)
            if new_stem != original_stem:
                renamed += 1
                tag = f" (renomeado de {original_stem})"
            else:
                tag = ""

            img_dst = img_dst_dir / f"{new_stem}{img_src.suffix.lower()}"
            lbl_dst = lbl_dst_dir / f"{new_stem}.txt"
            transfer(img_src, img_dst)
            transfer(lbl_src, lbl_dst)
            written += 1
            print(f"[{label}] {img_src.name} → {img_dst.name}{tag}")

    ingest(train_pairs, TRAIN_IMAGES, TRAIN_LABELS, "TRAIN")
    ingest(val_pairs, VAL_IMAGES, VAL_LABELS, "VAL  ")

    print("-" * 60)
    print(f"Pares processados: {written}")
    print(f"Pares renomeados por colisão: {renamed}")


if __name__ == "__main__":
    main()
