FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml README.md ./
RUN uv pip install --system --no-cache .

# Pre-download models at build time — eliminates cold-start latency
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-base')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY chaincheck/ ./chaincheck/

EXPOSE 10000
CMD ["sh", "-c", "uvicorn chaincheck.server:app --host 0.0.0.0 --port ${PORT:-10000}"]
