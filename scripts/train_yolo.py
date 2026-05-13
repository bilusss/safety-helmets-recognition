from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
LABELS_DIR = PROCESSED_DIR / "labels"
SPLIT_DIR = PROCESSED_DIR / "splits"
DATA_YAML = ROOT / "data" / "helmet.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model on prepared dataset.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--project", default="runs", help="Training output folder.")
    parser.add_argument("--name", default="helmet", help="Run name.")
    parser.add_argument("--device", default=None, help="Device to use, e.g. 0 or cpu.")
    return parser.parse_args()


def ensure_dirs() -> None:
    for sub in [
        SPLIT_DIR / "images" / "train",
        SPLIT_DIR / "images" / "val",
        SPLIT_DIR / "labels" / "train",
        SPLIT_DIR / "labels" / "val",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def iter_label_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        image_path = IMAGES_DIR / f"{label_path.stem}.jpg"
        if image_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def write_data_yaml() -> None:
    data_root = (PROCESSED_DIR / "splits").resolve()
    DATA_YAML.write_text(
        f"""
path: {data_root.as_posix()}
train: images/train
val: images/val
names:
  0: head
  1: helmet
""".lstrip()
    )


def split_dataset(pairs: list[tuple[Path, Path]], val_ratio: float, seed: int) -> None:
    if not pairs:
        raise RuntimeError("No processed labels found. Run dataset_preparing first.")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    split_idx = int(len(pairs) * (1 - val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    for image_path, label_path in train_pairs:
        link_or_copy(image_path, SPLIT_DIR / "images" / "train" / image_path.name)
        link_or_copy(label_path, SPLIT_DIR / "labels" / "train" / label_path.name)

    for image_path, label_path in val_pairs:
        link_or_copy(image_path, SPLIT_DIR / "images" / "val" / image_path.name)
        link_or_copy(label_path, SPLIT_DIR / "labels" / "val" / label_path.name)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    pairs = iter_label_pairs()
    split_dataset(pairs, args.val_ratio, args.seed)
    write_data_yaml()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
