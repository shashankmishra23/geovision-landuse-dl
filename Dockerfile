# CPU inference image for the API and dashboard (training is done on a GPU machine beforehand).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs configs
COPY src src
COPY app app
COPY pyproject.toml .
RUN pip install -e .

# The trained checkpoint and training reference stats are mounted/copied in at deploy time
COPY artifacts/best_model.pt artifacts/best_model.pt
COPY data/reference_stats.json data/reference_stats.json

RUN useradd --create-home appuser && mkdir -p monitoring && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
