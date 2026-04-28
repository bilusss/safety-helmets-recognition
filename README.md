# Safety Helmets Recognition

A computer vision project for detecting safety helmets and heads in images using YOLO.

## 📋 Table of Contents

- [Dataset Setup](#dataset-setup)
- [Data Format](#data-format)
- [Project Structure](#project-structure)

## Dataset Setup

### 1. Download Datasets

Download the following three datasets from the links below:

| Dataset | Size | Source | Rename to |
|---------|------|--------|-----------|
| Dataset 1 (GDUT-HWD) | 678.1 MB | [Google Drive](https://drive.google.com/file/d/1CLHnPfBVwwxmlUmz83pG0SZjc7k_A7Qw/view?usp=sharing) | `dataset1.zip` |
| Dataset 2 | 1.31 GB | [Kaggle - Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) | `dataset2.zip` |
| Dataset 3 | 4.58 GB | [Kaggle - Hardhat Vest Dataset v3](https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/code) | `dataset3.zip` |

> **Note:** Files downloaded from Kaggle are typically named `archive.zip`. Download both datasets and rename them accordingly the file size.

### 2. Organize Files

1. Move all downloaded zip files to the `data/raw/` directory
2. Rename files according to the table above

### 3. Run Dataset Setup

Run the following command in the project folder (root directory) to automatically process and organize the datasets:

```bash
make dataset_setup
```

This will extract, convert datasets to YOLO format, and organize all files in the `data/processed/` directory.

### Expected Directory Structure

```
data/
└── raw/
    ├── dataset1.zip (678.1 MB)
    ├── dataset2.zip (1.31 GB)
    └── dataset3.zip (4.58 GB)
```

## Data Format

### YOLO Format Requirement

All annotations must follow the YOLO format:

```
class_id x_center y_center width height
```

### Format Details

- **Coordinates must be normalized** (values between 0 and 1)
- **class_id = 0**: Head (without helmet)
- **class_id = 1**: Helmet (head with safety helmet)

### Example

```
0 0.5 0.5 0.3 0.4
1 0.7 0.6 0.25 0.35
```

## Project Structure

```
safety-helmets-recognition/
├── data/
│   ├── processed/
│   │   ├── images/
│   │   └── labels/
│   └── raw/
│       ├── dataset1.zip
│       ├── dataset2.zip
│       └── dataset3.zip
├── scripts/
│   └── dataset_setup.py
├── Makefile
├── pyproject.toml
└── README.md
```
