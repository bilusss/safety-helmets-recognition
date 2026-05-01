from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import hashlib
import shutil

import numpy as np
import pandas as pd


# paths 
ROOT       = Path(__file__).parent.parent
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
IMG_DIR    = PROC_DIR / "images"
LBL_DIR    = PROC_DIR / "labels"

# global counter so every image gets a unique 6-digit name
_counter: list[int] = [1]

# map of image hash -> first stored filename, used to avoid duplicates
_seen_hashes: dict[str, str] = {}
_skipped_duplicates: list[tuple[str, str]] = []

def _next_id() -> str:
    idx = _counter[0]
    _counter[0] += 1
    return f"{idx:06d}"


# helpers 

def ensure_dirs() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)


def dataset_unzipping() -> None:
    """Extract every zip in data/raw/ into a same-name sub-folder."""
    for zip_file in RAW_DIR.glob("*.zip"):
        dest = RAW_DIR / zip_file.stem
        if dest.exists():
            print(f"Already extracted: {zip_file.name}")
            continue
        try:
            with ZipFile(zip_file, "r") as z:
                z.extractall(dest)
            print(f"Extracted: {zip_file.name}")
        except Exception as e:
            print(f"Unzipping {zip_file.name} failed: {e}")


def _hash_image(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a stable hash of the image bytes for deduping."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_image(src: Path) -> Path | None:
    """Copy src image to IMG_DIR as <id>.jpg and return the label path."""
    img_hash = _hash_image(src)
    if img_hash in _seen_hashes:
        _skipped_duplicates.append((str(src), _seen_hashes[img_hash]))
        return None

    new_stem = _next_id()
    dst = IMG_DIR / f"{new_stem}.jpg"
    shutil.copy2(src, dst)
    _seen_hashes[img_hash] = dst.name
    return LBL_DIR / f"{new_stem}.txt"


def write_labels(label_path: Path, rows: list[str]) -> None:
    label_path.write_text("\n".join(rows) + "\n")


# Dataset 1 - GDUT-HWD (XML annotations)
# XML tag mapping: "none" -> 0 (head), anything else (blue/white/yellow/red) -> 1 (helmet)

XML_TO_CLASS = {
    "none": 0,
}

def _xml_class(name: str) -> int | None:
    """Return class id for a VOC-style object name, or None to skip."""
    name = name.strip().lower()
    if name == "none":
        return 0
    if name in ("blue", "white", "yellow", "red"):
        return 1
    return None


def _parse_voc_xml(xml_path: Path) -> tuple[int, int, list[str]]:
    """Return (img_width, img_height, yolo_rows)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    rows: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        cls = _xml_class(name)
        if cls is None:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        x_c = ((xmin + xmax) / 2) / img_w
        y_c = ((ymin + ymax) / 2) / img_h
        w   = (xmax - xmin) / img_w
        h   = (ymax - ymin) / img_h

        rows.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return img_w, img_h, rows


def process_dataset1() -> None:
    """Dataset 1 (GDUT-HWD): VOC XML annotations."""
    ds_root = RAW_DIR / "dataset1"
    if not ds_root.exists():
        print("Dataset 1 not found, skipping.")
        return

    ann_dir = ds_root / "Annotations"
    if not ann_dir.exists():
        # try nested structure
        candidates = list(ds_root.rglob("Annotations"))
        if candidates:
            ann_dir = candidates[0]
        else:
            print("Dataset 1: Annotations/ directory not found, skipping.")
            return

    # find image directory (usually JPEGImages or Images)
    img_src_dir = ann_dir.parent / "JPEGImages"
    if not img_src_dir.exists():
        img_src_dir = ann_dir.parent / "Images"
    if not img_src_dir.exists():
        img_src_dir = ann_dir.parent  # fallback: same level as Annotations

    processed = 0
    for xml_file in sorted(ann_dir.glob("*.xml")):
        # find matching image
        stem = xml_file.stem
        img_src = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = img_src_dir / (stem + ext)
            if candidate.exists():
                img_src = candidate
                break

        if img_src is None:
            print(f"Dataset 1: image for {xml_file.name} not found, skipping.")
            continue

        _, _, rows = _parse_voc_xml(xml_file)
        if not rows:
            continue  # skip images with no valid annotations

        lbl_path = copy_image(img_src)
        if lbl_path is None:
            continue

        write_labels(lbl_path, rows)
        processed += 1

    print(f"Dataset 1: processed {processed} images.")


# Dataset 2 - Hard Hat Detection (PASCAL VOC XML)
# helmet -> 1, head -> 0, person -> skip
# Skip images where every annotation is "person" (nothing useful left)

def _xml_class_ds2(name: str) -> int | None:
    name = name.strip().lower()
    if name == "helmet":
        return 1
    if name == "head":
        return 0
    # "person" and anything else -> skip
    return None


def _parse_voc_xml_ds2(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    rows: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        cls = _xml_class_ds2(name)
        if cls is None:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        x_c = ((xmin + xmax) / 2) / img_w
        y_c = ((ymin + ymax) / 2) / img_h
        w   = (xmax - xmin) / img_w
        h   = (ymax - ymin) / img_h

        rows.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return rows


def process_dataset2() -> None:
    """Dataset 2 (Hard Hat Detection): VOC XML, drop 'person'-only images."""
    ds_root = RAW_DIR / "dataset2"
    if not ds_root.exists():
        print("Dataset 2 not found, skipping.")
        return

    ann_dirs = list(ds_root.rglob("annotations")) + list(ds_root.rglob("Annotations"))
    if not ann_dirs:
        print("Dataset 2: annotations/ directory not found, skipping.")
        return

    processed = skipped = 0
    for ann_dir in ann_dirs:
        # find corresponding image folder
        img_src_dir = ann_dir.parent / "images"
        if not img_src_dir.exists():
            img_src_dir = ann_dir.parent / "JPEGImages"
        if not img_src_dir.exists():
            img_src_dir = ann_dir.parent

        for xml_file in sorted(ann_dir.glob("*.xml")):
            stem = xml_file.stem
            img_src = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = img_src_dir / (stem + ext)
                if candidate.exists():
                    img_src = candidate
                    break

            if img_src is None:
                continue

            rows = _parse_voc_xml_ds2(xml_file)
            if not rows:          # only persons (all stripped) -> skip
                skipped += 1
                continue

            lbl_path = copy_image(img_src)
            if lbl_path is None:
                continue

            write_labels(lbl_path, rows)
            processed += 1

    print(f"Dataset 2: processed {processed} images, skipped {skipped} person-only images.")


# Dataset 3 - Hardhat + Vest (YOLO labels already)
# Class remapping: 0 (hardhat) -> 1, 2 (head) -> 0, 1 (vest) and 3 (person) -> delete row
# Skip images where all kept rows would be empty (only vest / only person)

DS3_REMAP: dict[int, int | None] = {
    0: 1,    # hardhat  -> helmet (1)
    1: None, # vest     -> delete
    2: 0,    # head     -> head (0)
    3: None, # person   -> delete
}


def _remap_yolo_line(line: str) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None
    orig_cls = int(parts[0])
    new_cls = DS3_REMAP.get(orig_cls)
    if new_cls is None:
        return None
    return " ".join([str(new_cls)] + parts[1:])


def process_dataset3() -> None:
    """Dataset 3 (Hardhat + Vest): YOLO labels, remap classes, drop vest/person-only."""
    ds_root = RAW_DIR / "dataset3"
    if not ds_root.exists():
        print("Dataset 3 not found, skipping.")
        return

    # images and labels may live in train/valid/test sub-folders
    label_files = list(ds_root.rglob("*.txt"))
    if not label_files:
        print("Dataset 3: no .txt label files found, skipping.")
        return
    
    processed = skipped = 0
    for lbl_src in sorted(label_files):
        # skip the YOLO classes.txt file (not an annotation file)
        if lbl_src.name == "classes.txt":
            continue

        # find matching image: look next to the label, or in the parallel
        # images/<split>/ folder (dataset3 layout: dataset3/{images,labels}/<split>/)
        img_src = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = lbl_src.with_suffix(ext)
            if candidate.exists():
                img_src = candidate
                break
            # parallel images/<split>/ folder, e.g. dataset3/images/train/000001.jpg
            if len(lbl_src.parents) >= 3:
                parallel = lbl_src.parents[2] / "images" / lbl_src.parent.name / (lbl_src.stem + ext)
                if parallel.exists():
                    img_src = parallel
                    break

        if img_src is None:
            continue

        raw_lines = lbl_src.read_text().splitlines()
        remapped = [_remap_yolo_line(l) for l in raw_lines if l.strip()]
        rows = [r for r in remapped if r is not None]

        if not rows:
            skipped += 1
            continue

        lbl_path = copy_image(img_src)
        if lbl_path is None:
            continue

        write_labels(lbl_path, rows)
        processed += 1

    print(f"Dataset 3: processed {processed} images, skipped {skipped} vest/person-only images.")


# entry point

if __name__ == "__main__":
    ensure_dirs()
    dataset_unzipping()
    process_dataset1()
    process_dataset2()
    process_dataset3()
    if _skipped_duplicates:
        print(f"Skipped duplicate images: {len(_skipped_duplicates)}")
    print(f"\nDone. Total images written: {_counter[0] - 1}")