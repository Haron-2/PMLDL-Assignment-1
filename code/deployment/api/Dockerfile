FROM python:3.13-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt /app/code/deployment/api/requirements.txt
RUN pip install --no-cache-dir -r /app/code/deployment/api/requirements.txt

COPY code/deployment/api /app/code/deployment/api
COPY models/model.joblib /app/models/model.joblib

ENV MODEL_PATH=/app/models/model.joblib \
    MODEL_VERSION=rf-300-baseline

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/code/deployment/api"]
