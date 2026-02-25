"""
train_vball.py
---------------
Fine-tune do YOLOv8s para a classe única 'ball' em voleibol.

Uso:
    python train_vball.py --data data.yaml --batch 16
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def prepare_dataset() -> None:
    """Garante estrutura ./dataset/images/train e ./dataset/labels/train e move dados de img/train."""
    img_src = Path("img/train")
    lbl_src = img_src / "labels"
    img_dst = Path("dataset/images/train")
    lbl_dst = Path("dataset/labels/train")
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for jpg in img_src.glob("*.jpg"):
        target = img_dst / jpg.name
        if not target.exists():
            jpg.rename(target)
    for txt in lbl_src.glob("*.txt"):
        target = lbl_dst / txt.name
        if not target.exists():
            txt.rename(target)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8s para bola")
    p.add_argument("--data", type=str, default="data.yaml", help="Caminho para data.yaml")
    p.add_argument("--model", type=str, default="yolov8s.pt", help="Modelo base")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", type=str, default="vball-ball-only", help="Nome do experimento")
    p.add_argument("--device", type=str, default="0", help="GPU id (GTX 1650 -> 0)")
    return p.parse_args()


def main():
    prepare_dataset()
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml não encontrado em {data_path.resolve()}")

    model = YOLO(args.model)
    train_res = model.train(
    data='data.yaml',
    epochs=100,
    batch=2,        # Reduz de 16 para 2 (evita estouro de memória)
    imgsz=1280,     # Mantemos a resolução para a bola pequena
    device=0,
    workers=0,      # OBRIGATÓRIO: 0 para evitar o erro de DLL no Windows
    amp=False       # Mantém desativado para a GTX 1650
)

    # Copiar best.pt resultante para a raiz do projeto
    save_dir = Path(getattr(train_res, "save_dir", Path("runs/detect") / args.name))
    best_ckpt = save_dir / "weights" / "best.pt"
    if best_ckpt.exists():
        shutil.copy2(best_ckpt, Path("best.pt"))
        print(f"[INFO] best.pt copiado para {Path('best.pt').resolve()}")
    else:
        print(f"[AVISO] best.pt não encontrado em {best_ckpt}")


if __name__ == "__main__":
    main()
