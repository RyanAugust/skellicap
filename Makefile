# Makefile for Skellicap Pose Tracker

# Configuration
PYTHON = python3.12
VENV = venv
BIN = $(VENV)/bin
PIP = $(BIN)/pip
PY_VENV = $(BIN)/python
INPUT_VIDEO ?= 
OUTPUT_JSON ?= results.json
ANALYZED_JSON ?= analyzed_results.json
MIN_STRIDE_FRAMES ?= 10

.PHONY: all setup venv install process analyze clean

all: setup

setup: venv install

venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)

install: venv
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

process:
	@if [ -z "$(INPUT_VIDEO)" ]; then \
		echo "Error: INPUT_VIDEO is not set. Usage: make process INPUT_VIDEO=path/to/video.mp4 [OUTPUT_JSON=results.json]"; \
		exit 1; \
	fi
	@echo "Processing video: $(INPUT_VIDEO)"
	$(PY_VENV) -m skellicap.cli track --input "$(INPUT_VIDEO)" --output "$(OUTPUT_JSON)"

analyze:
	@if [ ! -f "$(OUTPUT_JSON)" ]; then \
		echo "Error: $(OUTPUT_JSON) not found. Run 'make process' first."; \
		exit 1; \
	fi
	@echo "Analyzing results from $(OUTPUT_JSON)..."
	$(PY_VENV) -m skellicap.cli analyze --input "$(OUTPUT_JSON)" --output "$(ANALYZED_JSON)" --min-stride-frames $(MIN_STRIDE_FRAMES)

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -f results.json
