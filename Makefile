.PHONY: dataset_preparing dataset_setup train analyze_video system_deps

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
	cd scripts && $(UV) run python analyze_video.py "$(SOURCE)"

# analyze a YouTube video (downloads first, then runs detection)
# usage: make analyze_youtube URL="https://www.youtube.com/watch?v=..."
analyze_youtube:
	cd scripts && $(UV) run python analyze_video.py "$(URL)"

system_deps: # install system libs required by OpenCV/Ultralytics
	sudo apt-get update && sudo apt-get install -y libgl1
