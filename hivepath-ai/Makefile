.PHONY: help install install-ml test test-fast lint run clean

help:
	@echo "install      Install the package and dev dependencies"
	@echo "install-ml   Also install torch for the neural service-time model"
	@echo "test         Run the full test suite"
	@echo "test-fast    Run everything except the slower solver tests"
	@echo "lint         Lint with ruff (if installed)"
	@echo "run          Start the API on \$$API_PORT (default 8000)"
	@echo "clean        Remove caches and build artefacts"

install:
	pip install -e ".[dev]"

install-ml:
	pip install -e ".[dev,ml,vlm]"

test:
	pytest

test-fast:
	pytest -m "not integration" --ignore=tests/test_solver.py

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check src tests || \
		echo "ruff not installed; pip install ruff"

run:
	python -m hivepath

clean:
	find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info
