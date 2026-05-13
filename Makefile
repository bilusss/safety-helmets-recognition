.PHONY: dataset_preparing dataset_setup train system_deps

PYTHON := python3
UV := uv

dataset_preparing: # organise dataset
	cd scripts && $(UV) run python dataset_preparing.py

dataset_setup: dataset_preparing

train: # train YOLO model (creates splits if missing)
	cd scripts && $(UV) run python train_yolo.py

system_deps: # install system libs required by OpenCV/Ultralytics
	sudo apt-get update && sudo apt-get install -y libgl1
