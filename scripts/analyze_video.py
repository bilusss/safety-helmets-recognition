from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = ROOT / "scripts" / "runs" / "detect" / "runs" / "helmet" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "outputs" / "video"


# class colors: head (no helmet) -> red, helmet -> green
CLASS_COLORS = {
    0: (0, 0, 255),    # head (BGR red)
    1: (0, 255, 0),    # helmet (BGR green)
}
CLASS_NAMES = {
    0: "head",
    1: "helmet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run helmet detection on a video file or YouTube URL."
    )
    parser.add_argument(
        "source",
        help="Path to a local video file or a YouTube URL.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained YOLO weights (.pt file).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video file path. Defaults to outputs/video/<source_stem>_detected.mp4.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (0.0-1.0).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: 0 for GPU, cpu for CPU.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=None,
        help="Maximum number of seconds to process (useful for long YouTube videos).",
    )
    return parser.parse_args()


def is_youtube_url(source: str) -> bool:
    return "youtube.com" in source or "youtu.be" in source


def ensure_yt_dlp() -> None:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("yt-dlp not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])


def download_youtube_video(url: str, dest_dir: Path) -> Path:
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(dest_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
        "merge_output_format": "mp4",
    }

    print(f"Downloading video from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "video")
        ext = info.get("ext", "mp4")

    downloaded = dest_dir / f"{video_id}.{ext}"
    if not downloaded.exists():
        # yt-dlp may have merged to .mp4 regardless of original ext
        downloaded = dest_dir / f"{video_id}.mp4"

    return downloaded


def draw_detections(frame, results, font_scale: float = 0.6, thickness: int = 2):
    import cv2

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # draw label background so text is readable
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return frame


def process_video(
    input_path: Path,
    output_path: Path,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str | None,
    max_duration: int | None,
) -> None:
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = int(max_duration * fps) if max_duration is not None else total_frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    predict_kwargs: dict = {
        "conf": conf,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device is not None:
        predict_kwargs["device"] = device

    frame_idx = 0
    print(f"Processing video: {input_path.name}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} fps")
    print(f"Total frames to process: {min(total_frames, max_frames)}")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        results = model.predict(frame, **predict_kwargs)
        annotated = draw_detections(frame, results)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / min(total_frames, max_frames) * 100
            print(f"  Frame {frame_idx}/{min(total_frames, max_frames)} ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"Output saved to: {output_path}")


def resolve_output_path(source: str, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)

    if is_youtube_url(source):
        stem = "youtube_video"
    else:
        stem = Path(source).stem

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{stem}_detected.mp4"


def main() -> None:
    args = parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        print(f"Model not found at: {model_path}")
        print("Train a model first with: make train")
        sys.exit(1)

    output_path = resolve_output_path(args.source, args.output)

    if is_youtube_url(args.source):
        ensure_yt_dlp()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = download_youtube_video(args.source, Path(tmp_dir))
            process_video(
                input_path=video_path,
                output_path=output_path,
                model_path=model_path,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                max_duration=args.max_duration,
            )
    else:
        video_path = Path(args.source)
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            sys.exit(1)
        process_video(
            input_path=video_path,
            output_path=output_path,
            model_path=model_path,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            max_duration=args.max_duration,
        )


if __name__ == "__main__":
    main()
