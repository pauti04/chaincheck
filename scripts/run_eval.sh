#!/usr/bin/env bash
set -euo pipefail

SAMPLES="${1:-500}"

chaincheck eval --method nli --samples "$SAMPLES" --output eval_results_nli.json
chaincheck eval --method consistency --samples "$SAMPLES" --output eval_results_consistency.json
chaincheck eval --method judge --samples "$SAMPLES" --output eval_results_judge.json
