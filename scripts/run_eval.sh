#!/usr/bin/env bash
# Run the full HaluEval benchmark for all detection methods.
set -euo pipefail

SAMPLES=${SAMPLES:-500}
OUTPUT_SUFFIX=${OUTPUT:-eval_results.json}

echo "==> NLI eval on $SAMPLES samples..."
chaincheck eval --method nli --samples "$SAMPLES" --output "nli_${OUTPUT_SUFFIX}"

echo "==> Judge eval on $SAMPLES samples..."
chaincheck eval --method judge --samples "$SAMPLES" --output "judge_${OUTPUT_SUFFIX}"

echo "==> Consistency eval on $SAMPLES samples..."
chaincheck eval --method consistency --samples "$SAMPLES" --output "consistency_${OUTPUT_SUFFIX}"

if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "==> Logprobs eval on $SAMPLES samples..."
    chaincheck eval --method logprobs --samples "$SAMPLES" --output "logprobs_${OUTPUT_SUFFIX}"
else
    echo "==> Skipping logprobs eval (OPENAI_API_KEY not set)"
fi

echo "==> All evals complete."
echo "    Results: nli_${OUTPUT_SUFFIX}, judge_${OUTPUT_SUFFIX}, consistency_${OUTPUT_SUFFIX}"
