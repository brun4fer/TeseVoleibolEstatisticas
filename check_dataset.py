"""
check_dataset.py
-----------------
Validação rápida do dataset YOLO:
- Verifica se imagens referenciadas pelos .txt existem.
- Confere formato: class x_center y_center width height (5 valores numéricos).

Uso:
    python check_dataset.py --data data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import yaml


def load_yaml(data_file: Path) -> dict:
    with open(data_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Valida dataset YOLO (imagens e labels)")
    p.add_argument("--data", type=str, default="data.yaml", help="Ficheiro data.yaml")
    return p.parse_args()


def iter_labels(label_dir: Path) -> List[Path]:
    return list(label_dir.rglob("*.txt"))


def validate_line(line: str) -> Tuple[bool, str]:
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"Esperados 5 valores, obtidos {len(parts)}"
    try:
        cls = int(float(parts[0]))
        vals = list(map(float, parts[1:]))
    except ValueError:
        return False, "Valores não numéricos"
    for v in vals:
        if v < 0 or v > 1:
            return False, f"Valor fora de [0,1]: {v}"
    if cls < 0:
        return False, f"Classe negativa: {cls}"
    return True, ""


def main():
    args = parse_args()
    data_file = Path(args.data)
    if not data_file.exists():
        print(f"[ERRO] data.yaml não encontrado: {data_file}")
        sys.exit(1)

    y = load_yaml(data_file)
    img_train = Path(y.get("train", ""))
    img_val = Path(y.get("val", ""))
    label_train = Path(y.get("names_path", y.get("labels", "")))  # suporte opcional

    errors = 0
    for split_name, img_root in [("train", img_train), ("val", img_val)]:
        if not img_root.exists():
            print(f"[ERRO] pasta de imagens do split '{split_name}' não existe: {img_root}")
            errors += 1
            continue
        lbl_root = Path(str(img_root).replace("images", "labels"))
        if not lbl_root.exists():
            print(f"[ERRO] pasta de labels não existe: {lbl_root}")
            errors += 1
            continue

        for lbl in iter_labels(lbl_root):
            img = img_root / lbl.relative_to(lbl_root).with_suffix(".jpg")
            if not img.exists():
                alt = img.with_suffix(".png")
                if alt.exists():
                    img = alt
                else:
                    print(f"[ERRO] Imagem não encontrada para {lbl}: {img}")
                    errors += 1
                    continue
            with open(lbl, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    ok, msg = validate_line(line)
                    if not ok:
                        print(f"[ERRO] {lbl}:{idx} -> {msg}")
                        errors += 1
    if errors == 0:
        print("[OK] Dataset validado sem erros.")
    else:
        print(f"[FIM] Encontrados {errors} problemas.")
        sys.exit(1)


if __name__ == "__main__":
    main()
