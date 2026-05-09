FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml README.md ./
RUN uv pip install --system --no-cache .

# Pre-download NLI model at build time — eliminates cold-start latency.
# Default: nli-MiniLM2-L6-H768 (~90 MB, fits in 512 MB free-tier hosting).
# For higher accuracy swap to cross-encoder/nli-deberta-v3-base (~700 MB)
# by passing --build-arg NLI_MODEL=cross-encoder/nli-deberta-v3-base
ARG NLI_MODEL=cross-encoder/nli-MiniLM2-L6-H768
ENV CHAINCHECK_NLI_MODEL=${NLI_MODEL}
RUN python -c "import os; from sentence_transformers import CrossEncoder; CrossEncoder(os.environ['CHAINCHECK_NLI_MODEL'])"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY chaincheck/ ./chaincheck/

EXPOSE 10000
CMD ["sh", "-c", "uvicorn chaincheck.server:app --host 0.0.0.0 --port ${PORT:-10000}"]
