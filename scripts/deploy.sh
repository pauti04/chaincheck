#!/usr/bin/env bash
# Deploy ChainCheck to Railway.
set -euo pipefail

if ! command -v railway &>/dev/null; then
    echo "Railway CLI not found. Install with: npm install -g @railway/cli"
    exit 1
fi

echo "==> Deploying ChainCheck to Railway..."
railway up --detach

echo "==> Deploy triggered. Monitor at: https://railway.app/dashboard"
