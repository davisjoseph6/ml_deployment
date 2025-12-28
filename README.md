# PCBQC ML Deployment (YOLO Segmentation + MobileSAM) — `ml_deployment`

This repository contains a containerized (Docker) and Kubernetes (AKS) deployment for a PCB quality-control inference service.  
It exposes a REST API for:

- **Segmentation / inference** (YOLO segmentation model + MobileSAM post-processing)
- **Health & readiness checks**
- **Local retraining trigger** (`/api/retrain`) running **inside the web container** (Option B)

> Current deployment mode: **Option B (local retrain in `pcbqc-web`)**  
> No Azure Storage / AML secrets required for the deployed service.

---

## Repository layout

High-level structure (as seen in the project root):

- `app/` — application source (FastAPI/Uvicorn/Gunicorn)
- `ai_model_artifacts/` — model weights and artifacts (YOLO + MobileSAM, configs)
- `data/` — datasets / sample data used for testing or local runs
- `runs/` — training/inference outputs (Ultralytics runs)
- `k8s/` — Kubernetes manifests
  - `k8s/pcbqc-web.yaml` — web API Deployment + Service
  - `k8s/pcbqc-web-ingress.yaml` — ingress routing
  - `k8s/retrain-worker.yaml` — (legacy / optional) worker manifest (Option A)
  - `k8s/instancetypes/` — AKS instance type helpers
- `deploy/` — deployment utilities/scripts
- `ops/` — operational scripts and notes
- `Dockerfile` — CPU image
- `Dockerfile.gpu` — GPU image (optional)
- `docker-compose.yml` — local dev/run helper
- `swagger.json` — OpenAPI/Swagger export
- `test_images/` — sample inputs for local testing
- `scripts/` — convenience scripts (requests, test harness, etc.)

---

## Features

### Web API
- Runs a FastAPI app behind Gunicorn + Uvicorn worker(s)
- Provides:
  - `GET /health` — liveness/readiness + runtime info
  - `POST /api/retrain` — starts local retraining inside the web container
  - (Additional inference endpoints exist depending on `app/` implementation)

### Local retraining (Option B)
- Retraining is executed **inside the `pcbqc-web` container**, using Ultralytics.
- No external queue worker is needed.
- This repo previously explored “Option A” (Azure Queue + separate retrain worker),
  but final state uses **Option B**.

---

## Requirements

### Local development
- Python 3.10+ (recommended)
- `docker` and `docker compose`
- `curl` (for API testing)

### Kubernetes / AKS
- `kubectl`
- Access to a Kubernetes cluster (AKS in the final setup)
- Ingress controller already present (AKS Web App Routing or equivalent)

---

## Environment variables

### Required for running
- `YOLO_CONFIG_DIR=/tmp/Ultralytics`  
  Ensures Ultralytics settings directory is writable in container environments.

### Optional (implementation-dependent)
The application may support additional environment variables depending on code under `app/`.
Check `app/` for configuration defaults.

---

## Quick start (Docker)

### 1) Build (CPU image)
```bash
docker build -t pcbqc-web:local -f Dockerfile .
```

### 2) Run
```bash
docker run --rm -it \
  -p 8000:8000 \
  -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
  pcbqc-web:local
```

### 3) Test
```bash
curl -fsS http://127.0.0.1:8000/health | jq .
curl -sS -X POST http://127.0.0.1:8000/api/retrain
```

## Quick start (docker-compose)
If your docker-compose.yml is configured for the web service:
```bash
docker compose up --build
```

Test:
```bash
curl -fsS http://127.0.0.1:8000/health | jq .
curl -sS -X POST http://127.0.0.1:8000/api/retrain
```

## Deploy to Kubernetes (AKS) — Option B
These steps deploy only the web API + ingress.

### 1) Apply manifests
```bash
kubectl -n pcbqc apply -f k8s/pcbqc-web.yaml
kubectl -n pcbqc apply -f k8s/pcbqc-web-ingress.yaml
```

### 2) Wait for rollout
```bash
kubectl -n pcbqc rollout status deploy/pcbqc-web
kubectl -n pcbqc get pods -l app=pcbqc-web -o wide
```

### 3) Port-forward (local testing)
```bash
kubectl -n pcbqc port-forward --address 127.0.0.1 svc/pcbqc-web-svc 8080:80
```

Test:
```bash
curl -fsS http://127.0.0.1:8080/health | jq .
curl -sS -X POST http://127.0.0.1:8080/api/retrain
```

### 4) Ingress (cluster endpoint)
Check ingress:
```bash
kubectl -n pcbqc get ingress pcbqc-web -o wide
```

Health check via hostname:
```bash
curl -fsS http://<YOUR_INGRESS_HOST>/health | jq .
```

## Important: avoiding the legacy Option A components
This repo includes k8s/retrain-worker.yaml as a legacy/experimental manifest (Option A).
Option B does not require it.

To avoid accidentally recreating Option A:

- Do not run kubectl apply -f k8s/ blindly
- Apply only:
  - k8s/pcbqc-web.yaml
  - k8s/pcbqc-web-ingress.yaml

## API usage
```bash
curl -fsS http://127.0.0.1:8000/health | jq .
```

Example response fields (may vary):

- status
- version
- torch_device_env
- cuda_available
- yolo_model_path
- sam_model_path


## Retrain (local)
```bash
curl -sS -X POST http://127.0.0.1:8000/api/retrain
```

Typical response:
```bash
{"ok": true, "message": "Training started"}
```

Logs can be viewed from:

- Docker: docker logs -f <container>
- Kubernetes:
```bash
kubectl -n pcbqc logs -f deploy/pcbqc-web --tail=200
```

## GPU build (optional)

If you have a GPU-enabled environment:
```bash
docker build -t pcbqc-web:gpu -f Dockerfile.gpu .
```

Run similarly (port mapping may vary):
```bash
docker run --rm -it --gpus all -p 8000:8000 pcbqc-web:gpu
```

## Troubleshooting
### curl: (7) Failed to connect ...

- Your kubectl port-forward is not running, or died.
- Re-run:
```bash
kubectl -n pcbqc port-forward --address 127.0.0.1 svc/pcbqc-web-svc 8080:80
```

### Unable to listen on port 8080: address already in use
- Another kubectl port-forward is still running and holding the port.
- Find and kill it:
```bash
ps aux | grep 'kubectl.*port-forward' | grep 8080
kill <PID>
```

### Pod stuck in CreateContainerConfigError

- Usually caused by missing secrets referenced in env vars.
- For Option B, ensure pcbqc-web env contains only YOLO_CONFIG_DIR:
- kubectl -n pcbqc get deploy pcbqc-web -o jsonpath='{.spec.template.spec.containers[0].env}' | jq .

## Security notes

- Do not commit secrets or credentials.
- Keep .env out of git (already in .gitignore in most setups).
- Prefer Kubernetes Secrets / external secret managers for any production credentials.

License

Internal / project-specific. Add a LICENSE file if this repository is intended for distribution.

## Acknowledgements

- Ultralytics (YOLO)
- PyTorch
- FastAPI + Uvicorn + Gunicorn
- MobileSAM
- Kubernetes / AKS

```bash
::contentReference[oaicite:0]{index=0}
```
