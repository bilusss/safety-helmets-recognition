.PHONY: dataset_preparing

PYTHON := python3
UV := uv

dataset_preparing: # organise dataset
	cd scripts && $(UV) run python dataset_preparing.py
