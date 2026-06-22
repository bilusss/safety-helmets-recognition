# Safety Helmets Recognition

A computer vision project for detecting safety helmets and bare heads in images and videos using YOLOv8.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Video Analysis](#video-analysis)
- [Data Format](#data-format)
- [Project Structure](#project-structure)
- [Training Results](#training-results)

## Requirements

- Python 3.12 or newer
- [uv](https://github.com/astral-sh/uv) package manager
- GPU with CUDA is recommended for training and video inference; CPU works but is significantly slower

**Linux only:** OpenCV requires a system library for GUI/video support:

```bash
sudo apt-get update && sudo apt-get install -y libgl1
# or via Makefile:
make system_deps
```

**Windows:** no additional system dependencies are required.

## Installation

### Linux

```bash
git clone https://github.com/your-org/safety-helmets-recognition.git
cd safety-helmets-recognition
uv sync
```

### Windows

Install uv from https://github.com/astral-sh/uv, then:

```powershell
git clone https://github.com/your-org/safety-helmets-recognition.git
cd safety-helmets-recognition
uv sync
```

`make` is not available on Windows by default. All `make` commands below have a direct `uv run` equivalent listed alongside them.

## Dataset Setup

### 1. Download Datasets

Download the following three datasets:

| Dataset | Size | Source | Rename to |
|---------|------|--------|-----------|
| Dataset 1 (GDUT-HWD) | 678.1 MB | [Google Drive](https://drive.google.com/file/d/1CLHnPfBVwwxmlUmz83pG0SZjc7k_A7Qw/view?usp=sharing) | `dataset1.zip` |
| Dataset 2 | 1.31 GB | [Kaggle - Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) | `dataset2.zip` |
| Dataset 3 | 4.58 GB | [Kaggle - Hardhat Vest Dataset v3](https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/code) | `dataset3.zip` |

> Files downloaded from Kaggle are typically named `archive.zip`. Rename them by file size as shown above.

### 2. Organize Files

Move all zip files into `data/raw/`.

### 3. Run Dataset Preparation

Linux:

```bash
make dataset_preparing
```

Windows:

```powershell
cd scripts
uv run python dataset_preparing.py
```

This extracts the archives, converts all annotations to YOLO format, deduplicates images by SHA-256 hash, and writes everything to `data/processed/images/` and `data/processed/labels/`.

At minimum, dataset1 is required. dataset2 and dataset3 are optional; place them in `data/raw/` and rerun the preparation step before training.

## Training

### Linux

```bash
make train
```

### Windows

```powershell
cd scripts
uv run python train_yolo.py
```

### Optional Arguments

```bash
uv run python scripts/train_yolo.py --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16
```

The script creates a train/val split (80/20) under `data/processed/splits/`, writes `data/helmet.yaml`, and starts YOLOv8 training. Weights are saved to `scripts/runs/detect/runs/helmet/weights/`.

## Video Analysis

The `analyze_video.py` script runs helmet detection on a local video file, a YouTube URL, or a webcam. Detections are drawn as bounding boxes: green for helmets, red for bare heads.

It runs in two modes:

- **Batch (default):** processes the whole video and writes an annotated `.mp4` to `outputs/video/`.
- **Live (`--show`):** opens a window and plays the video back **in real time** with detections drawn on each frame, an FPS counter, and a red `NO HELMET` banner whenever a bare head is detected. Press `q` or `ESC` to stop. This is the real-time mode for demos and webcams.

Since YOLOv8n inference is ~0.5 ms/frame on GPU, live playback keeps up with normal video frame rates; the preview is paced to the source FPS so a file plays at natural speed.

### Linux

Analyze a local file:

```bash
make analyze_video SOURCE=path/to/video.mp4
```

Analyze a YouTube video:

```bash
make analyze_youtube URL="https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

### Windows

Analyze a local file:

```powershell
cd scripts
uv run python analyze_video.py path\to\video.mp4
```

Analyze a YouTube video:

```powershell
cd scripts
uv run python analyze_video.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

### Live Preview (Real Time)

Add `--show` to watch the detection live in a window instead of waiting for an output file.

Linux:

```bash
# local file, live window (also saves the annotated file)
make show_video SOURCE=path/to/video.mp4

# YouTube video, live window
make show_youtube URL="https://www.youtube.com/watch?v=XXXXXXXXXXX"

# webcam, live window only (no file written); CAM defaults to 0
make webcam
make webcam CAM=1
```

Windows / direct `uv run`:

```powershell
cd scripts
# local file, live preview only (no file written)
uv run python analyze_video.py path\to\video.mp4 --show --no-save

# webcam index 0, live preview
uv run python analyze_video.py 0 --show --no-save
```

`--show` opens a live window; `--no-save` skips writing the output file (use it for a pure preview). Without `--show`, the script runs in batch mode and only writes the output file, as before. Press `q` or `ESC` in the window to stop early.

### All Options

```
usage: analyze_video.py [-h] [--model MODEL] [--output OUTPUT]
                        [--conf CONF] [--imgsz IMGSZ] [--device DEVICE]
                        [--max-duration MAX_DURATION] [--show] [--no-save]
                        source

positional arguments:
  source                Path to a local video file, a YouTube URL, or a webcam index (e.g. 0).

options:
  --model MODEL         Path to trained YOLO weights. Default: best.pt from the last training run.
  --output OUTPUT       Output file path. Default: outputs/video/<name>_detected.mp4.
  --conf CONF           Detection confidence threshold (default: 0.25).
  --imgsz IMGSZ         Inference image size (default: 640).
  --device DEVICE       Device: 0 for first GPU, cpu for CPU.
  --max-duration MAX_DURATION
                        Stop after this many seconds. Useful for long YouTube videos.
  --show                Display annotated frames live in a window, in real time (q/ESC to quit).
  --no-save             Do not write an output file (use with --show for a pure live preview).
```

### Examples

Process only the first 2 minutes of a YouTube video using GPU 0:

```bash
uv run python scripts/analyze_video.py \
  "https://www.youtube.com/watch?v=XXXXXXXXXXX" \
  --max-duration 120 \
  --device 0
```

Run on CPU with a lower confidence threshold:

```bash
uv run python scripts/analyze_video.py footage.mp4 --device cpu --conf 0.4
```

YouTube videos are downloaded at up to 720p to a temporary directory, which is deleted automatically after processing. The output video is saved as standard MP4.

## Data Format

All annotations use the YOLO format:

```
class_id x_center y_center width height
```

Coordinates are normalized to the range [0, 1].

- `class_id = 0`: head (no helmet)
- `class_id = 1`: helmet

Example:

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
│   │   ├── labels/
│   │   └── splits/
│   └── raw/
│       ├── dataset1.zip
│       ├── dataset2.zip
│       └── dataset3.zip
├── outputs/
│   └── video/
├── scripts/
│   ├── analyze_video.py
│   ├── dataset_preparing.py
│   └── train_yolo.py
├── Makefile
├── pyproject.toml
└── README.md
```

## Training Results

Results from a 50-epoch run on the full three-dataset combination:

```
Model: YOLOv8n   |   Runtime: 1.263 h   |   Device: NVIDIA GeForce RTX 5070 Ti

Class     Images   Instances   P       R       mAP50   mAP50-95
all        5891     45452      0.927   0.896   0.940   0.574
head       1822     27207      0.921   0.892   0.931   0.512
helmet     4667     18245      0.933   0.901   0.949   0.636

Speed: 0.5 ms inference per image
```
