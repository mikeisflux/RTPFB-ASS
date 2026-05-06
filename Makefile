.PHONY: help install dev third-party models test lint fmt mypy ci docker run clean

help:
	@echo "Targets:"
	@echo "  install       pip install -e ."
	@echo "  dev           pip install -e .[dev,test,preset,lipsync,voice]"
	@echo "  third-party   clone upstream repos into third_party/"
	@echo "  models        print where to put model weights"
	@echo "  test          pytest (skips gpu/audio/camera markers)"
	@echo "  lint          ruff check + format check"
	@echo "  fmt           ruff format + ruff --fix"
	@echo "  mypy          mypy src/rtpfb"
	@echo "  ci            lint + mypy + test"
	@echo "  docker        docker build -t rtpfb ."
	@echo "  run           rtpfb run --preset presets/streaming-knnvc.yaml --target assets/face.png"
	@echo "  clean         remove build artefacts and caches"

install:
	pip install -e .

dev:
	pip install -e ".[dev,test,preset,lipsync,voice]"

third-party:
	bash scripts/install_third_party.sh

models:
	bash scripts/download_models.sh

test:
	pytest -v -m "not gpu and not audio and not camera and not integration"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

fmt:
	ruff check --fix src/ tests/
	ruff format src/ tests/

mypy:
	mypy src/rtpfb

ci: lint mypy test

docker:
	docker build -t rtpfb .

run:
	rtpfb run --preset presets/streaming-knnvc.yaml --target assets/face.png

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
