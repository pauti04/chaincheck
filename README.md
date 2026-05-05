# ChainCheck

[![PyPI](https://img.shields.io/pypi/v/chaincheck)](https://pypi.org/project/chaincheck/)
[![CI](https://github.com/pauti04/chaincheck/actions/workflows/ci.yml/badge.svg)](https://github.com/pauti04/chaincheck/actions)
[![Coverage](https://codecov.io/gh/pauti04/chaincheck/branch/main/graph/badge.svg)](https://codecov.io/gh/pauti04/chaincheck)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Claim-level hallucination detection for LLM outputs.** Achieves **79% F1 / 94% precision** on HaluEval-QA (n=500, gpt-4o-mini judge). Give ChainCheck a response and optional source context, and it tells you exactly which claims are unsupported — not just whether the whole response is bad.

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
│   decompose()         │  gpt-4o-mini → JSON array of atomic claims
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

**HaluEval QA** — response-level (n=500, balanced, 50% hallucinated, with reference context):

| Method      | Precision | Recall | F1        | ECE ↓  | Avg Latency | P95 Latency |
|-------------|-----------|--------|-----------|--------|-------------|-------------|
| NLI         | 0.810     | 0.444  | 0.574     | 0.279  | 55 ms       | 97 ms       |
| Judge (+ second pass) | **0.944** | 0.676  | **0.788** | 0.161  | 1125 ms     | 2045 ms     |
| Consistency | 0.000     | 0.000  | 0.000     | 0.500  | 2117 ms     | 4740 ms     |
| Logprobs    | 0.263     | 0.084  | 0.127     | —      | 1401 ms     | 2859 ms     |
| **NLI+Judge ensemble** | — | — | **0.741** | — | ~55–1199 ms | — |

> Ensemble F1 on held-out 20% of HaluEval; weights tuned via Nelder-Mead on training 80%.
> Consistency predicts "not hallucinated" for all samples (F1=0, ECE=0.5); excluded from default ensemble.
> ECE — lower is better; 0 = perfectly calibrated confidence scores.

**TruthfulQA generation** — response-level (n=500, adversarial questions, no reference context):

| Method | Precision | Recall | F1        | ECE ↓  | Avg Latency | P95 Latency |
|--------|-----------|--------|-----------|--------|-------------|-------------|
| Judge (fact-check mode) | **0.744** | 0.664  | **0.702** | 0.202  | 2822 ms     | 4608 ms     |

> No reference context — fact-check mode uses a world-knowledge prompt with today's date injected
> and confidence capped at 0.7. F1 improved from 0.683 → 0.702 vs the old generic judge prompt.
> Precision gain (+8.4pp) reflects the stricter fact-check framing; recall trade-off is expected
> since the model is more conservative without a source document.

**HaluEval claim-level** — discrimination metrics (n=100 pairs, NLI, no annotation required):

| Metric | Value | Meaning |
|--------|-------|---------|
| Clean flagging rate ↓ | 0.127 | 12.7% of claims in correct responses incorrectly flagged |
| Halluc flagging rate ↑ | 0.525 | 52.5% of claims in hallucinated responses flagged |
| Discrimination ratio ↑ | **4.13×** | hallucinated responses have 4× more flagged claims |
| Claim AUC ↑ | **0.913** | per-claim NLI scores vs response-level labels |
| Avg claims / response | 0.8 | decomposition quality proxy (short answers → fewer claims) |

> Claim AUC of 0.913 means NLI's per-claim scores rank claims from hallucinated responses above
> claims from correct responses 91.3% of the time — without any claim-level human annotation.
> Avg claims/response of 0.8 reflects a known limitation: the decomposer produces fewer claims
> for terse answers; longer factual responses yield richer claim-level signal.

> Full results: [`nli_eval_results.json`](nli_eval_results.json), [`judge_eval_results.json`](judge_eval_results.json), [`truthfulqa_judge_eval_results.json`](truthfulqa_judge_eval_results.json), [`claimlevel_nli_eval_results.json`](claimlevel_nli_eval_results.json).
> Reproduce with `bash scripts/run_eval.sh`. Results committed weekly by the [eval workflow](.github/workflows/eval.yml).

---

## Quick start

```bash
pip install chaincheck
export OPENAI_API_KEY=sk-...
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

Each line of `inputs.jsonl` is a JSON object with `response` and optionally `context`:
```json
{"response": "The Eiffel Tower was built in 1887.", "context": "It was completed in 1889."}
{"response": "Water boils at 90°C at sea level.", "context": ""}
```

**Cascade mode (34× faster on clear-cut cases):**
```bash
chaincheck check \
  --response "..." --context "..." \
  --cascade
# runs NLI first (51 ms); escalates to judge only when score is 0.2–0.8
```

**Debug claim decomposition:**
```bash
chaincheck check --response "..." --context "..." --debug-claims
# prints extracted atomic claims before the scoring table
# useful for diagnosing why a known hallucination was missed
```

**Start the API server:**
```bash
chaincheck serve --port 8000
# → http://localhost:8000/docs
```

---

## Which method should I use?

| Situation | Recommended |
|-----------|-------------|
| You have a context/reference document (RAG) | `--methods nli,judge` (default) |
| High throughput, latency < 100 ms per check | `--methods nli` |
| Fast LLM check without NLI model download | `--methods qa` |
| You want to flag borderline cases for human review | `--cascade` — runs NLI first, judge only on 0.2–0.8 scores |
| Checking open-ended generation with no ground truth | `--methods consistency` |
| Need the highest-precision signal (95.5%) | `--methods judge` |

**QA method** (`--methods qa`) — asks the LLM "does the context support this claim? yes/no" at temperature=0. No chain-of-thought, ~3× fewer output tokens than judge. Useful when you want an LLM check but don't want to download the NLI model, or as a fast second opinion.

**Consistency** detects when a model gives *inconsistent* answers to the same question. It scores F1=0.000 on HaluEval (confidently wrong models are consistently wrong). Use it only for open-ended generation with no reference context — it is not a substitute for NLI or judge on context-grounded tasks.

**Logprobs** requires a prompt and is most useful as a cheap pre-filter: high token uncertainty correlates with hallucination risk but does not catch confident errors.

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

**POST /stream** — same body as `/check`, returns `text/event-stream`. Events arrive as each method completes:
```
data: {"type": "claims",  "claims": ["...", "..."], "request_id": "uuid"}
data: {"type": "method",  "method": "nli",   "score": 0.12, "latency_ms": 210}
data: {"type": "method",  "method": "judge",  "score": 0.08, "latency_ms": 1340}
data: {"type": "result",  "data": {...full DetectionResult...}}
data: [DONE]
```

**POST /batch** — same as `/check` but body is `{"inputs": [...]}`, returns array.

**GET /history?limit=20** — returns the last N detection results (max 100), persisted in SQLite:
```json
[{
  "request_id": "uuid",
  "created_at": 1714900000.0,
  "response_preview": "The Eiffel Tower is located in...",
  "aggregate_score": 0.71,
  "risk_level": "high",
  "total_latency_ms": 1420.3,
  "methods": ["judge", "nli"]
}]
```

**GET /health**
```json
{ "status": "ok", "version": "0.6.0", "models_loaded": true }
```

---

## Configuration

All settings via environment variables:

| Variable                | Default                    | Description                              |
|-------------------------|----------------------------|------------------------------------------|
| `OPENAI_API_KEY`        | —                          | Required — used for all LLM calls by default |
| `ANTHROPIC_API_KEY`     | —                          | Optional — set to use Claude models instead |
| `OLLAMA_BASE_URL`       | `http://localhost:11434`   | Optional — prefix model IDs with `ollama:` to use local models |
| `JUDGE_MODEL`           | `gpt-4o-mini`              | Judge LLM model ID                       |
| `CONSISTENCY_MODEL`     | `gpt-4o-mini`              | Model for self-consistency sampling      |
| `DECOMPOSE_MODEL`       | `gpt-4o-mini`              | Model for claim decomposition            |
| `CONSISTENCY_SAMPLES`   | `5`                        | LLM samples per consistency check        |
| `NLI_BATCH_SIZE`        | `16`                       | Claims per NLI inference batch           |
| `CACHE_PATH`            | `.chaincheck_cache`        | diskcache directory (24h TTL; key = SHA-256 of the full response string) |
| `NLI_THRESHOLD`         | `0.5`                      | Min confidence to label a claim          |
| `CONSISTENCY_THRESHOLD` | `0.82`                     | Min similarity to consider consistent    |
| `RISK_LOW_THRESHOLD`    | `0.3`                      | Aggregate score below this → "low"       |
| `RISK_HIGH_THRESHOLD`   | `0.7`                      | Aggregate score at or above this → "high"|
| `NLI_WEIGHT`            | `0.10`                     | NLI weight — Nelder-Mead tuned on 80% HaluEval, held-out F1=0.741 |
| `CONSISTENCY_WEIGHT`    | `0.0`                      | Consistency disabled in ensemble (F1=0.168 on factual tasks) |
| `JUDGE_WEIGHT`          | `0.60`                     | Judge weight — dominant signal, precision=0.965 |
| `LOGPROB_WEIGHT`        | `0.30`                     | Logprobs weight — useful secondary signal in ensemble |
| `LOGPROB_MODEL`         | `gpt-4o-mini`              | OpenAI model for logprobs method         |
| `LOGPROB_THRESHOLD`     | `-1.5`                     | Token log-prob below this → uncertain    |

---

## What's measured vs what's claimed

ChainCheck's pitch is *claim-level* detection: tell you which specific sentence is wrong, not just whether the whole response is bad. The benchmarks above measure *response-level* F1 — whether the pipeline's aggregate score correctly labels the full response as hallucinated or not. These are related but not the same thing.

To bridge this gap, ChainCheck ships a dedicated claim-level evaluation:

```bash
chaincheck eval --dataset halueval-claims --method nli --samples 100 --output claims.json
```

This uses HaluEval pairs (each question has both a correct answer and a hallucinated answer against the same context) and reports:
- **Clean flagging rate** — fraction of claims in *correct* responses that get incorrectly flagged (claim-level false positive rate)
- **Halluc flagging rate** — fraction of claims in *hallucinated* responses that get flagged (claim-level coverage)
- **Discrimination ratio** — halluc / clean; a ratio of 3 means hallucinated responses have 3× more flagged claims
- **Claim AUC** — AUC of per-claim scores against response-level labels; no claim-level annotation required

Exact claim-level precision/recall requires human-annotated atomic facts (as in [FactScore](https://arxiv.org/abs/2305.14251)) and is on the roadmap. The metrics above are a principled proxy and characterise claim-level behaviour in a way no other hallucination detection benchmark currently reports.

---

## What we learned

**NLI and judge complement each other.** NLI has high precision (0.810) at 51 ms — fast and conservative, rarely cries wolf. Judge has even higher precision (0.965) — when it flags something as hallucinated, it's right 96.5% of the time. NLI is 34× faster, making it ideal for high-throughput filtering before running the more accurate judge on borderline cases.

**Self-consistency does not transfer to factual benchmarks.** Consistency F1 is 0.168 on HaluEval — below random (accuracy 0.228). This is expected: the method detects when a model gives *inconsistent* answers to the same question, but a confidently wrong model is consistently wrong. Consistency is most useful for detecting knowledge gaps (open-ended questions the model hallucinates answers to), not for catching facts that contradict a provided context.

**Latency is the real cost, not the accuracy.** NLI is 34× faster than judge (51 ms vs 1755 ms) with lower but still useful F1. In a high-throughput serving context, running NLI on every request and reserving judge for borderline cases (0.3–0.7 score) cuts average latency by ~34× while keeping precision above 0.80.

**Cascade cuts average latency by up to 34× on clear-cut cases.** Running NLI first (51 ms) and escalating to judge only when the score is in the 0.2–0.8 ambiguous band avoids the 1755 ms judge call for responses that are obviously clean or obviously hallucinated. Enable with `--cascade` on the CLI or `cascade=True` in the Python API.

**Confidence calibration (ECE) is now measured.** ECE (Expected Calibration Error) measures whether a score of 0.9 actually means "90% likely to be hallucinated." Lower ECE = more trustworthy confidence numbers. Run `chaincheck eval` and check the ECE column to see how well-calibrated each method's scores are.

**Claim decomposition quality is the hidden variable.** Both NLI and judge score individual claims — if decompose() merges two facts into one claim, a partially-wrong claim can still pass. The decomposition quality (measured by claim count per sentence) directly bounds downstream F1 ceiling. Logprobs F1 (0.127) reflects this: token-level uncertainty alone is not sufficient signal without claim-level grounding.

---

## Deployment (Railway)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Set secrets
railway variables set OPENAI_API_KEY=sk-...

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
