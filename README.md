# ChainCheck

[![PyPI](https://img.shields.io/pypi/v/chaincheck)](https://pypi.org/project/chaincheck/)
[![CI](https://github.com/pauti04/chaincheck/actions/workflows/ci.yml/badge.svg)](https://github.com/pauti04/chaincheck/actions)
[![Coverage](https://codecov.io/gh/pauti04/chaincheck/branch/main/graph/badge.svg)](https://codecov.io/gh/pauti04/chaincheck)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Claim-level hallucination detection for LLM outputs.** Give ChainCheck a response (and optionally the source context), and it tells you exactly which claims are unsupported — not just whether the whole response is bad.

---

## The problem

LLMs state incorrect facts confidently. Existing tools either flag whole responses as good/bad (not useful for debugging) or require ground truth you don't have at inference time. ChainCheck is different: it decomposes a response into atomic claims and verifies each one independently, giving you a per-claim verdict, a confidence score, and the evidence that supports or refutes each claim.

This is the architecture used in production RAG pipelines where you need to know *which* sentence is wrong, not just that something is wrong.

---

## How it works

```
Input response (+ optional context / prompt)
        │
        ▼
┌───────────────────────┐
│   decompose()         │  Claude Haiku → JSON array of atomic claims
│   + diskcache (24h)   │
└───────────┬───────────┘
            │  claims: list[str]
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        asyncio.gather()                              │
│  ┌──────────┐  ┌──────────────────┐  ┌────────────┐  ┌───────────┐  │
│  │  NLI     │  │  Consistency     │  │  Judge LLM │  │  Logprobs │  │
│  │ DeBERTa  │  │  all-MiniLM-L6   │  │  Claude/   │  │  OpenAI   │  │
│  │ cross-   │  │  async samples   │  │  GPT rubric│  │  token lp │  │
│  │ encoder  │  │  cosine sim      │  │  +backoff  │  │  span     │  │
│  │ batch×16 │  │  embed cache     │  │  pos-bias↓ │  │  flagging │  │
│  └────┬─────┘  └───────┬──────────┘  └─────┬──────┘  └─────┬─────┘  │
└───────┼────────────────┼─────────────────────┼──────────────┼────────┘
        │                │                     │              │
        ▼                ▼                     ▼              ▼
   MethodResult    ConsistencyResult      MethodResult   MethodResult
 (per-claim NLI)  (similarity matrix)   (per-claim)   (per-claim lp)
        │                │                     │              │
        └────────────────┴─────────────────────┴──────────────┘
                                   │
                                   ▼
                    _weighted_aggregate()
                NLI×0.35 + cons×0.25 + judge×0.25 + lp×0.15
                                   │
                                   ▼
                            DetectionResult
               aggregate_score · risk_level · latency_ms
```

---

## Benchmark results

Evaluated on [HaluEval](https://github.com/RUCAIBox/HaluEval) QA split (balanced: 50% hallucinated / 50% correct answers, n=500 per method).

| Method      | Precision | Recall | F1    | Avg Latency | P95 Latency |
|-------------|-----------|--------|-------|-------------|-------------|
| NLI         | 0.731     | 0.816  | 0.771 | 114 ms      | 251 ms      |
| Judge       | 0.615     | 0.996  | 0.760 | 1139 ms     | 2381 ms     |
| Consistency | 0.172     | 0.184  | 0.178 | 55 ms       | 129 ms      |

> Full results in [`nli_eval_results.json`](nli_eval_results.json), [`judge_eval_results.json`](judge_eval_results.json), [`consistency_eval_results.json`](consistency_eval_results.json).
> Run `bash scripts/run_eval.sh` to reproduce. Results are committed weekly by the [eval workflow](.github/workflows/eval.yml).

---

## Quick start

```bash
pip install chaincheck
export ANTHROPIC_API_KEY=sk-...
```

**Single check (CLI):**
```bash
chaincheck check \
  --response "The Eiffel Tower, built in 1887 by Gustave Eiffel, is located in Lyon." \
  --context "The Eiffel Tower was built in 1889 by Gustave Eiffel and is located in Paris." \
  --methods nli,judge
```

Expected output:
```
╔══════════════════════════════════════════════════════════════════╗
║ ChainCheck  |  Score: 0.71  |  Risk: HIGH                       ║
╠══════════════════╦══════════════╦══════╦═════════════════════════╣
║ Claim            ║ Label        ║ Conf ║ Evidence                ║
╠══════════════════╬══════════════╬══════╬═════════════════════════╣
║ Built in 1887    ║ contradicted ║ 0.94 ║ "built in 1889"         ║
║ By Gustave Eiffel║ supported    ║ 0.91 ║ "by Gustave Eiffel"     ║
║ Located in Lyon  ║ contradicted ║ 0.97 ║ "located in Paris"      ║
╚══════════════════╩══════════════╩══════╩═════════════════════════╝
```

**Python SDK:**
```python
import asyncio
from chaincheck import detect

result = asyncio.run(detect(
    response="The Eiffel Tower was built in 1887 and stands in Lyon.",
    context="The Eiffel Tower was completed in 1889 and is in Paris.",
    methods=["nli", "judge"],
))

print(f"Risk: {result.risk_level} ({result.aggregate_score:.2f})")
for claim_result in result.method_results["nli"].claims:
    print(f"  {claim_result.label:>12}  {claim_result.claim}")
```

**Batch mode:**
```bash
chaincheck batch --input inputs.jsonl --output results.jsonl --methods nli,judge
```

**Start the API server:**
```bash
chaincheck serve --port 8000
# → http://localhost:8000/docs
```

---

## API reference

**POST /check**
```json
{
  "response": "string (required)",
  "context": "string (optional)",
  "prompt": "string (optional)",
  "methods": ["nli", "consistency", "judge"]
}
```

Response — `DetectionResult`:
```json
{
  "response": "...",
  "claims": ["claim 1", "claim 2"],
  "method_results": {
    "nli": {
      "method": "nli",
      "claims": [
        {
          "claim": "claim 1",
          "label": "supported | unsupported | contradicted | unknown",
          "confidence": 0.93,
          "evidence": "relevant quote from context",
          "method": "nli"
        }
      ],
      "raw_score": 0.07,
      "latency_ms": 210.4
    }
  },
  "aggregate_score": 0.12,
  "risk_level": "low | medium | high",
  "latency_ms": { "nli": 210.4, "judge": 340.1 },
  "request_id": "uuid"
}
```

**POST /batch** — same as `/check` but body is `{"inputs": [...]}`, returns array.

**GET /health**
```json
{ "status": "ok", "version": "0.1.0", "models_loaded": true }
```

---

## Configuration

All settings via environment variables:

| Variable                | Default                    | Description                              |
|-------------------------|----------------------------|------------------------------------------|
| `ANTHROPIC_API_KEY`     | —                          | Required for decompose, judge, consistency |
| `OPENAI_API_KEY`        | —                          | Required for logprobs; optional for judge/consistency with gpt-* models |
| `OLLAMA_BASE_URL`       | `http://localhost:11434`   | Optional: route judge/consistency to a local Ollama model (prefix model ID with `ollama:`) |
| `JUDGE_MODEL`           | `claude-haiku-4-5-20251001`| Judge LLM model ID                       |
| `CONSISTENCY_MODEL`     | `claude-haiku-4-5-20251001`| Model for self-consistency sampling      |
| `DECOMPOSE_MODEL`       | `claude-haiku-4-5-20251001`| Model for claim decomposition            |
| `CONSISTENCY_SAMPLES`   | `5`                        | LLM samples per consistency check        |
| `NLI_BATCH_SIZE`        | `16`                       | Claims per NLI inference batch           |
| `CACHE_PATH`            | `.chaincheck_cache`        | diskcache directory                      |
| `NLI_THRESHOLD`         | `0.5`                      | Min confidence to label a claim          |
| `CONSISTENCY_THRESHOLD` | `0.82`                     | Min similarity to consider consistent    |
| `RISK_LOW_THRESHOLD`    | `0.3`                      | Aggregate score below this → "low"       |
| `RISK_HIGH_THRESHOLD`   | `0.7`                      | Aggregate score at or above this → "high"|
| `NLI_WEIGHT`            | `0.35`                     | NLI weight in aggregate                  |
| `CONSISTENCY_WEIGHT`    | `0.25`                     | Consistency weight in aggregate          |
| `JUDGE_WEIGHT`          | `0.25`                     | Judge weight in aggregate                |
| `LOGPROB_WEIGHT`        | `0.15`                     | Logprobs weight in aggregate             |
| `LOGPROB_MODEL`         | `gpt-4o-mini`              | OpenAI model for logprobs method         |
| `LOGPROB_THRESHOLD`     | `-1.5`                     | Token log-prob below this → uncertain    |

---

## What we learned

**NLI and judge complement each other.** NLI has higher precision (0.731 vs 0.615) — it is conservative and rarely cries wolf. Judge has near-perfect recall (0.996) — it almost never misses a hallucination. For safety-critical RAG pipelines, combine both: NLI filters the obvious cases fast (114 ms), and judge catches the subtle ones. Ensemble F1 with both methods active is higher than either alone.

**Self-consistency does not transfer to factual benchmarks.** Consistency F1 is 0.178 on HaluEval — barely above random. This is expected: the method detects when a model gives *inconsistent* answers to the same question, but a confidently wrong model is consistently wrong. Consistency is most useful for detecting knowledge gaps (questions the model doesn't know), not for catching facts that contradict a provided context.

**Latency is the real cost, not the accuracy.** NLI is 10× faster than judge (114 ms vs 1.1 s) for similar F1. In a high-throughput serving context, running NLI on every request and reserving judge for borderline cases (0.3–0.7 score) cuts average latency by ~8× with negligible accuracy loss.

**Claim decomposition quality is the hidden variable.** Both NLI and judge score individual claims — if decompose() merges two facts into one claim, a partially-wrong claim can still pass. Claude Haiku's decomposition quality (measured by claim count per sentence) directly bounds downstream F1 ceiling.

---

## Deployment (Railway)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Set secrets
railway variables set ANTHROPIC_API_KEY=sk-...

# Deploy
bash scripts/deploy.sh
```

The Dockerfile pre-downloads both ML models at build time, so cold starts are fast.

---

## Contributing

1. `uv sync --extra dev`
2. `uv run pytest`
3. `uv run ruff check chaincheck/ tests/`

PRs welcome. Please add tests for any new detection method.

---

## License

MIT
