#!/usr/bin/env bash
# Run the full benchmark suite: HaluEval (response-level) + claim-level + TruthfulQA.
set -euo pipefail

SAMPLES=${SAMPLES:-500}
CLAIM_PAIRS=${CLAIM_PAIRS:-100}
OUTPUT_SUFFIX=${OUTPUT:-eval_results.json}

echo "==> HaluEval — NLI ($SAMPLES samples)..."
uv run chaincheck eval --method nli --samples "$SAMPLES" --output "nli_${OUTPUT_SUFFIX}"

echo "==> HaluEval — Judge ($SAMPLES samples)..."
uv run chaincheck eval --method judge --samples "$SAMPLES" --output "judge_${OUTPUT_SUFFIX}"

echo "==> HaluEval — Consistency ($SAMPLES samples)..."
uv run chaincheck eval --method consistency --samples "$SAMPLES" --output "consistency_${OUTPUT_SUFFIX}"

if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "==> HaluEval — Logprobs ($SAMPLES samples)..."
    uv run chaincheck eval --method logprobs --samples "$SAMPLES" --output "logprobs_${OUTPUT_SUFFIX}"
else
    echo "==> Skipping logprobs eval (OPENAI_API_KEY not set)"
fi

echo "==> Claim-level discrimination — NLI ($CLAIM_PAIRS pairs)..."
uv run chaincheck eval --dataset halueval-claims --method nli \
    --samples "$CLAIM_PAIRS" --output "claimlevel_nli_${OUTPUT_SUFFIX}"

echo "==> TruthfulQA — Judge (200 samples)..."
uv run chaincheck eval --dataset truthfulqa --method judge \
    --samples 200 --output "truthfulqa_judge_${OUTPUT_SUFFIX}"

echo "==> All evals complete."
echo "    HaluEval:    nli_${OUTPUT_SUFFIX}  judge_${OUTPUT_SUFFIX}  consistency_${OUTPUT_SUFFIX}"
echo "    Claim-level: claimlevel_nli_${OUTPUT_SUFFIX}"
echo "    TruthfulQA:  truthfulqa_judge_${OUTPUT_SUFFIX}"
