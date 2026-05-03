FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY chaincheck ./chaincheck

RUN uv pip install --system .

RUN python -c "from chaincheck.methods.nli import preload_model as nli; from chaincheck.methods.consistency import preload_model as emb; nli(); emb()"

EXPOSE 8000

CMD ["uvicorn", "chaincheck.server:app", "--host", "0.0.0.0", "--port", "8000"]
