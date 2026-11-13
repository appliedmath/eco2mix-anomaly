# API image for Kubernetes: the normal API image + a read-only snapshot of the
# champion's results (models/) and the processed data.
# Why a snapshot: the pods are stateless replicas; in Docker Compose the API
# mounts ./models and ./data, in Kubernetes there is no shared project folder.
# New champion or new data -> rebuild this image -> rolling update (k8s/deploy.sh).
# In production the model would come from the MLflow registry and the data
# from object storage (S3/MinIO) instead.
#
# Build (after `docker compose build api`):
#   docker build -f docker/api-k8s.Dockerfile -t eco2mix-api-k8s:latest .
FROM eco2mix-anomaly-api:latest

COPY data/processed/consumption_30min.parquet /app/data/processed/consumption_30min.parquet
COPY models/ /app/models/
