.PHONY: install test lint serve docker-build docker-up eval

install:
	uv sync --extra dev

test:
	uv run pytest --tb=short -q

lint:
	uv run ruff check chaincheck/ tests/
	uv run ruff format --check chaincheck/ tests/

serve:
	chaincheck serve --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up

eval-nli:
	chaincheck eval --method nli --samples 500 --output nli_eval_results.json

eval-judge:
	chaincheck eval --method judge --samples 500 --output judge_eval_results.json

eval-all: eval-nli eval-judge
	@echo "Eval complete."
