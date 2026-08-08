.PHONY: help install lint format test test-fast run-cli clean

help:        ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n",$$1,$$2}'

install:     ## Install all deps via uv
	uv sync

lint:        ## Ruff check
	uv run ruff check .

format:      ## Ruff format + autofix
	uv run ruff format .
	uv run ruff check --fix .

test:        ## Pytest with coverage
	uv run pytest --cov=lucidata --cov-report=term-missing

test-fast:   ## Pytest excluding slow (benchmark) tests
	uv run pytest -m "not slow"

run-cli:     ## Smoke-run the CLI
	uv run lucidata --help

clean:       ## Remove build artifacts
	rm -rf dist build .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +