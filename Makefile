.PHONY: dataset_preparing dataset_setup train analyze_video analyze_youtube show_video show_youtube webcam system_deps

PYTHON := python3
UV := uv

dataset_preparing: # organise dataset
	cd scripts && $(UV) run python dataset_preparing.py

dataset_setup: dataset_preparing

train: # train YOLO model (creates splits if missing)
	cd scripts && $(UV) run python train_yolo.py

# analyze a local video file
# usage: make analyze_video SOURCE=path/to/video.mp4
analyze_video:
	$(UV) run python scripts/analyze_video.py "$(SOURCE)"

# analyze a YouTube video (downloads first, then runs detection)
# usage: make analyze_youtube URL="https://www.youtube.com/watch?v=..."
analyze_youtube:
	$(UV) run python scripts/analyze_video.py "$(URL)"

# live real-time preview of a local video file (q or ESC to quit)
# usage: make show_video SOURCE=path/to/video.mp4
show_video:
	$(UV) run python scripts/analyze_video.py "$(SOURCE)" --show

# live real-time preview of a YouTube video (downloads first)
# usage: make show_youtube URL="https://www.youtube.com/watch?v=..."
show_youtube:
	$(UV) run python scripts/analyze_video.py "$(URL)" --show

# live real-time detection from a webcam, preview only (no file written)
# usage: make webcam            (or: make webcam CAM=1 for a second camera)
webcam:
	$(UV) run python scripts/analyze_video.py "$(or $(CAM),0)" --show --no-save

system_deps: # install system libs required by OpenCV/Ultralytics
	sudo apt-get update && sudo apt-get install -y libgl1
