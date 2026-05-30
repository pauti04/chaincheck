# Changelog

## [0.6.0] — 2025-05-05

### Added
- **Fact-check mode** — when no context is provided, the judge switches to a world-knowledge prompt with today's date injected; confidence capped at 0.7 to prevent overconfidence without a source document
- **Second-pass recheck** — `supported` claims with confidence < 0.45 are re-examined with a stricter prompt; improves recall without hurting precision
- **NLI source attribution** — each claim result now carries `source_id` and `source_url` pointing to the document that most strongly entailed or contradicted it
- **Inline document annotation UI** — pass multiple `documents` objects; the UI highlights which source document backs each claim
- **Analytics dashboard** — scan history with aggregate score distribution and method latency charts
- **Clickable history panel** — re-open any past result from the sidebar
- **`cascade` field in `/check` API** — run NLI first, escalate to judge only on borderline scores (0.2–0.8); up to ~19× average latency reduction on clear-cut cases

### Changed
- Scoring formula fixed: `mean(bad_confidence) / n_claims` instead of `bad_weight / total_weight` (old formula gave artificially high scores on low-confidence verdicts)
- Weights re-tuned via Nelder-Mead on 80% HaluEval holdout: NLI×0.10, Judge×0.60
- Judge model pinned to `gpt-4o-mini` (gpt-4o reasons more carefully and finds nuance, lowering recall on this benchmark)

### Benchmark
- Judge: **F1=0.763, Precision=0.936** on HaluEval-QA n=500
- TruthfulQA fact-check: **F1=0.702, Precision=0.744** (no reference context)
- Claim AUC: **0.913** on HaluEval claim-level pairs

---

## [0.5.0] — 2025-04

### Added
- Anthropic judge support alongside OpenAI — set `ANTHROPIC_API_KEY` to switch
- Proxy mode — `chaincheck serve` can forward to an upstream LLM via `PROXY_URL`
- Feedback endpoint — thumbs up/down on any result stored in SQLite
- Source attribution UI — per-claim badges showing which document provided evidence

---

## [0.4.0] — 2025-04

### Added
- Real SSE streaming — `/stream` endpoint yields per-method events as they complete
- SQLite history — `/history` endpoint returns last N scans with scores and latencies
- Redesigned UI — light theme, 3-phase analysis animation, SVG gauge, JetBrains Mono

---

## [0.3.0] — 2025-04

### Added
- QA method — LLM yes/no claim verification, ~3× fewer output tokens than judge
- Cascade mode (`--cascade`) — NLI-first with judge escalation on ambiguous band
- Cascade Pareto analysis — `chaincheck eval --dataset cascade` sweeps (low, high) thresholds
- Claim-level evaluation — discrimination ratio, claim AUC, clean/halluc flagging rates
- TruthfulQA benchmark support — `chaincheck eval --dataset truthfulqa`
- 35 new unit tests (181 total, 72% → 85% coverage)

---

## [0.2.0] — 2025-03

### Added
- Ensemble weight tuning — ECE calibration, Nelder-Mead optimised weights
- `--debug-claims` flag — print decomposed atomic claims before scoring
- Decision guide in README — which method to use in which situation
- Logprobs method — token-level uncertainty via OpenAI log probabilities

---

## [0.1.0] — 2025-03

### Added
- Initial release — 4-method hallucination detection: NLI, consistency, judge, logprobs
- `chaincheck check`, `batch`, `eval`, `serve`, `compare` CLI commands
- FastAPI server with `/check`, `/stream`, `/batch`, `/health` endpoints
- HaluEval benchmark runner
- Docker + Railway deployment config
