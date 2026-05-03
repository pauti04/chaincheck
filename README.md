# ChainCheck

[![PyPI](https://img.shields.io/pypi/v/chaincheck.svg)](https://pypi.org/project/chaincheck/)
[![CI](https://github.com/yourusername/chaincheck/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/chaincheck/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/yourusername/chaincheck/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/chaincheck)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Claim-level hallucination detection for LLM responses using NLI, self-consistency, and LLM-as-judge signals.

## Problem

LLMs often answer with confident, fluent statements that are wrong, contradicted by the retrieved source, or simply unsupported. That failure mode is especially costly in retrieval-augmented generation, research assistants, legal and medical workflows, and internal knowledge tools where a single unsupported sentence can make an otherwise useful answer unsafe to trust.

Most hallucination checks still return one response-level score. ChainCheck works at the claim level: it decomposes a response into atomic factual assertions, checks each assertion against source context, and returns evidence, confidence, and aggregate risk. The goal is not just to say "this answer is bad"; it is to show exactly which claims need attention and why.

## How It Works

```text
response + optional prompt/context
             |
             v
      atomic decomposition
             |
             v
   +---------+----------+
   |         |          |
   v         v          v
  NLI   consistency   judge
   |         |          |
   +---------+----------+
             |
             v
        weighted score
             |
             v
      DetectionResult JSON
```

## Benchmark Results

Run `scripts/run_eval.sh` to reproduce these numbers on HaluEval. The scheduled GitHub Actions eval workflow updates the checked-in result JSON files weekly.

| Method | Precision | Recall | F1 | Avg Latency |
| --- | ---: | ---: | ---: | ---: |
| NLI | 0.731 | 0.816 | 0.771 | 114.4 ms |
| Consistency | 0.172 | 0.184 | 0.178 | 54.8 ms |
| Judge | 0.615 | 0.996 | 0.760 | 1139.2 ms |

## Quick Start

```bash
pip install chaincheck
```

Check one response:

```bash
chaincheck check \
  --response "Paris is the capital of Germany." \
  --context "Paris is the capital of France. Berlin is the capital of Germany." \
  --methods nli,judge
```

Expected output:

```text
ChainCheck: HIGH risk (0.78)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Claim                            ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Paris is the capital of Germany. │ high   │
└──────────────────────────────────┴────────┘
```

Batch mode:

```bash
chaincheck batch --input input.jsonl --output results.jsonl --methods all
```

Run a benchmark:

```bash
chaincheck eval --method nli --samples 500 --output eval_results_nli.json
```

## API Reference

Start the server:

```bash
chaincheck serve --port 8000 --reload
```

`POST /check`

```json
{
  "response": "Paris is the capital of Germany.",
  "context": "Paris is the capital of France. Berlin is the capital of Germany.",
  "prompt": "What is the capital of Germany?",
  "methods": ["nli", "judge"]
}
```

Response:

```json
{
  "response": "Paris is the capital of Germany.",
  "claims": ["Paris is the capital of Germany."],
  "methods": {
    "nli": [
      {
        "claim": "Paris is the capital of Germany.",
        "label": "contradicted",
        "confidence": 0.91,
        "evidence": "Berlin is the capital of Germany."
      }
    ]
  },
  "aggregate_score": 0.78,
  "risk_level": "high",
  "latency_ms": {"nli": 42.1}
}
```

`POST /batch`

```json
{
  "inputs": [
    {
      "response": "The Eiffel Tower is in Paris.",
      "context": "The Eiffel Tower is in Paris.",
      "methods": ["nli"]
    }
  ]
}
```

`GET /health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "models_loaded": {"nli": true, "embedding": true}
}
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | unset | Required for default Claude decomposition, judge, and consistency sampling. |
| `OPENAI_API_KEY` | unset | Optional for `gpt-*` judge models. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Optional local Ollama endpoint. |
| `JUDGE_MODEL` | `claude-haiku-4-5` | Judge model. Use `gpt-4o-mini` or `ollama:model-name` to switch providers. |
| `CONSISTENCY_MODEL` | `claude-haiku-4-5` | Model used for self-consistency samples. |
| `CONSISTENCY_SAMPLES` | `5` | Number of parallel sampled responses. |
| `NLI_BATCH_SIZE` | `16` | NLI batch size. |
| `CACHE_PATH` | `.chaincheck_cache` | Disk cache location. |
| `NLI_THRESHOLD` | `0.5` | Minimum NLI confidence before forcing a neutral label. |
| `CONSISTENCY_THRESHOLD` | `0.82` | Mean similarity threshold below which a response is inconsistent. |
| `RISK_LOW_THRESHOLD` | `0.3` | Aggregate score below this is low risk. |
| `RISK_HIGH_THRESHOLD` | `0.7` | Aggregate score at or above this is high risk. |
| `WEIGHT_NLI` | `0.4` | Aggregate weight for NLI. |
| `WEIGHT_CONSISTENCY` | `0.3` | Aggregate weight for self-consistency. |
| `WEIGHT_JUDGE` | `0.3` | Aggregate weight for judge. |

## What We Learned

On the first 500 HaluEval QA examples, NLI is the strongest standalone method by F1 while staying fast enough for interactive use. The judge path has extremely high recall but lower precision, which makes it useful as a conservative safety layer when missing hallucinations is more expensive than reviewing false positives. Self-consistency is fast after embeddings are cached, but performs poorly as a standalone hallucination detector on this benchmark; it is better treated as a weak auxiliary signal than a primary classifier.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

Run the full benchmark:

```bash
scripts/run_eval.sh 500
```

Build and run Docker:

```bash
docker build -t chaincheck .
docker run -p 8000:8000 --env ANTHROPIC_API_KEY chaincheck
```

## Contributing

Issues and pull requests are welcome. Please keep changes typed, tested, and covered by `ruff` and `pytest`.

## License

MIT. See [LICENSE](LICENSE).
