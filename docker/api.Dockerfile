# API service: serves the stored detections, never trains
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1. Dependencies first: this layer is cached until the file changes
COPY docker/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2. Then the code (changes often, so it comes last)
COPY src/ src/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
