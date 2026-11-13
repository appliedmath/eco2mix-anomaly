#!/usr/bin/env bash
# Deploy the API to the local Kubernetes cluster (Docker Desktop).
#   bash k8s/deploy.sh          build the snapshot image + apply everything
#   bash k8s/deploy.sh delete   remove the namespace (and all its objects)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "delete" ]]; then
  kubectl delete namespace eco2mix --ignore-not-found
  exit 0
fi

echo "1/4 image: API + snapshot of models/ and the processed data"
docker compose build api
docker build -q -f docker/api-k8s.Dockerfile -t eco2mix-api-k8s:latest .

echo "2/4 namespace + settings"
kubectl apply -f k8s/namespace.yml -f k8s/configmap.yml

echo "3/4 secret from .env (values are never printed)"
API_KEY=$(grep '^API_KEY=' .env | cut -d= -f2-)
AIRFLOW_PASSWORD=$(grep '^AIRFLOW_ADMIN_PASSWORD=' .env | cut -d= -f2-)
kubectl -n eco2mix create secret generic eco2mix-api-secrets \
  --from-literal=API_KEY="$API_KEY" --from-literal=AIRFLOW_PASSWORD="$AIRFLOW_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
unset API_KEY AIRFLOW_PASSWORD

echo "4/4 deployment + service, then a rolling restart to pick up the new image"
kubectl apply -f k8s/deployment.yml -f k8s/service.yml
kubectl -n eco2mix rollout restart deployment/eco2mix-api
kubectl -n eco2mix rollout status deployment/eco2mix-api --timeout=180s
kubectl -n eco2mix get pods,svc
echo "API on http://localhost:8001/health"
