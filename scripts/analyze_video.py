from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
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

WINDOW_NAME = "Safety Helmet Detection - press q or ESC to quit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run helmet detection on a video file, YouTube URL, or webcam."
    )
    parser.add_argument(
        "source",
        help="Path to a local video file, a YouTube URL, or a webcam index (e.g. 0).",
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
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames live in a window, in real time.",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not write an output video file (use with --show for a pure live preview).",
    )
    parser.set_defaults(save=True)
    return parser.parse_args()


def is_youtube_url(source: str) -> bool:
    return "youtube.com" in source or "youtu.be" in source


def is_webcam(source: str) -> bool:
    return source.isdigit()


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
    """Draw bounding boxes and return per-class detection counts for this frame."""
    import cv2

    counts = {0: 0, 1: 0}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            counts[cls_id] = counts.get(cls_id, 0) + 1
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

    return frame, counts


def draw_hud(frame, head_count: int, helmet_count: int, fps: float):
    """Overlay a live status line and a warning banner when a bare head is present."""
    import cv2

    h, w = frame.shape[:2]

    status = f"FPS: {fps:5.1f}   helmets: {helmet_count}   no-helmet: {head_count}"
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(
        frame, status, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # the assignment is about detecting MISSING helmets -> make it loud
    if head_count > 0:
        banner = f"!! NO HELMET: {head_count} !!"
        (tw, th), bl = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
        cx = max(10, (w - tw) // 2)
        cv2.rectangle(frame, (cx - 12, 40), (cx + tw + 12, 40 + th + bl + 16), (0, 0, 255), -1)
        cv2.putText(
            frame, banner, (cx, 40 + th + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA,
        )

    return frame


def process_video(
    input_source: str | int,
    output_path: Path,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str | None,
    max_duration: int | None,
    show: bool,
    save: bool,
) -> None:
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)

    is_cam = isinstance(input_source, int)
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {input_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0  # webcams often report 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_duration is not None:
        max_frames: int | None = int(max_duration * fps)
    elif total_frames > 0:
        max_frames = total_frames
    else:
        max_frames = None  # webcam / unknown length -> run until stopped

    writer = None
    if save:
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

    # pace playback to the source frame rate only when previewing a finite file
    target_dt = (1.0 / fps) if (show and not is_cam) else 0.0

    print(f"Processing source: {input_source}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} fps")
    if max_frames:
        print(f"Total frames to process: {max_frames}")
    if show:
        print("Live preview: press 'q' or ESC in the window to stop.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fps_ema = fps
    frame_idx = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, **predict_kwargs)
            annotated, counts = draw_detections(frame, results)

            proc_dt = time.perf_counter() - t0
            inst_fps = 1.0 / proc_dt if proc_dt > 0 else 0.0
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps
            annotated = draw_hud(annotated, counts.get(0, 0), counts.get(1, 0), fps_ema)

            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow(WINDOW_NAME, annotated)
                remaining = target_dt - (time.perf_counter() - t0)
                wait_ms = max(1, int(remaining * 1000)) if target_dt > 0 else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    print("Stopped by user.")
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed.")
                    break

            frame_idx += 1
            if not show and max_frames and frame_idx % 50 == 0:
                pct = frame_idx / max_frames * 100
                print(f"  Frame {frame_idx}/{max_frames} ({pct:.1f}%)")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if writer is not None:
        print(f"Output saved to: {output_path}")


def resolve_output_path(source: str, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)

    if is_youtube_url(source):
        stem = "youtube_video"
    elif is_webcam(source):
        stem = "webcam"
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

    if not args.show and not args.save:
        print("Nothing to do: pass --show to preview live, or drop --no-save to write a file.")
        sys.exit(1)

    output_path = resolve_output_path(args.source, args.output)

    common = dict(
        output_path=output_path,
        model_path=model_path,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        max_duration=args.max_duration,
        show=args.show,
        save=args.save,
    )

    if is_webcam(args.source):
        process_video(input_source=int(args.source), **common)
    elif is_youtube_url(args.source):
        ensure_yt_dlp()
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = download_youtube_video(args.source, Path(tmp_dir))
            process_video(input_source=str(video_path), **common)
    else:
        video_path = Path(args.source)
        if not video_path.exists():
            # also try relative to the project root, so paths work regardless
            # of the current working directory (e.g. when run from scripts/)
            alt = ROOT / args.source
            if alt.exists():
                video_path = alt
            else:
                print(f"Video file not found: {args.source}")
                sys.exit(1)
        process_video(input_source=str(video_path), **common)


if __name__ == "__main__":
    main()
