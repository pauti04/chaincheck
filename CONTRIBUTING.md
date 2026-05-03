# Contributing to ChainCheck

## Setup

```bash
git clone https://github.com/yourusername/chaincheck
cd chaincheck
uv sync --extra dev
```

## Running tests

```bash
uv run pytest                        # all tests with coverage
uv run pytest tests/test_nli.py -v   # single file
```

Coverage must stay above 80%. New detection methods require tests.

## Adding a detection method

1. Create `chaincheck/methods/your_method.py` — implement `async def check_your_method(claims, context) -> MethodResult`
2. Export it from `chaincheck/methods/__init__.py`
3. Add it to `_METHOD_WEIGHTS` in `chaincheck/detect.py` and wire into `detect()`
4. Add `--method your_method` support in `chaincheck/eval/halueval.py`
5. Write tests in `tests/test_your_method.py`

## Code style

```bash
uv run ruff check chaincheck/ tests/   # must pass with zero warnings
uv run ruff check --fix chaincheck/ tests/  # auto-fix most issues
```

Rules: type hints on every function, docstrings on every public function, no function longer than 40 lines, no bare `except`.

## Running benchmarks locally

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...          # optional, for logprobs method

bash scripts/run_eval.sh              # all methods, 500 samples each
```

Results are written to `*_eval_results.json` and committed weekly by the eval workflow.

## Pull requests

- Keep PRs focused: one method, one feature, or one fix per PR
- Update the README benchmark table if your change affects detection accuracy
- CI must pass (ruff + pytest) before merging
