# CPU inference service. Build:  docker build -t sed-api .
# Run:    docker run --rm -p 8000:8000 -v $PWD/outputs/baseline/best:/model sed-api
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/model HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# optional extra CA certificates (corporate proxies); see docker/certs/README.md
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

WORKDIR /app
RUN pip install --index-url ${TORCH_INDEX_URL} torch
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "sed.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
