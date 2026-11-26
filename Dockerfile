FROM python:3.13-slim
WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY triageguard ./triageguard
COPY artifacts/models/baseline_bundle.joblib ./artifacts/models/baseline_bundle.joblib
EXPOSE 8000
CMD ["uvicorn", "triageguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
