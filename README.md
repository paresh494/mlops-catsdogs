# MLOps Pipeline — Cats vs Dogs Binary Image Classification

End-to-end MLOps pipeline for a pet-adoption platform's Cats-vs-Dogs classifier:
model development + experiment tracking, packaging + containerization, a CI
pipeline that tests and publishes images, a CD pipeline that deploys and
smoke-tests, and post-deployment monitoring.

> Course: **S1-25 AIMLCZG523 — MLOps**, Assignment 2.

---

## TL;DR — run the whole thing locally

```bash
# 0. install
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .

# 1. get data  (use the real Kaggle set OR the synthetic smoke set)
python scripts/generate_synthetic_data.py --per-class 150 --out data/raw
#   real dataset: unzip Kaggle "Dogs vs Cats" into data/raw/ (cats/ & dogs/ or cat.*.jpg / dog.*.jpg)

# 2. preprocess -> 224x224 RGB, split 80/10/10
python -m mlops_catsdogs.data.preprocess

# 3. train baseline CNN + log to MLflow
python -m mlops_catsdogs.train --epochs 5 --batch-size 32
mlflow ui --backend-store-uri sqlite:///mlflow.db      # inspect runs at :5000

# 4. serve
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -F "file=@some_pet.jpg;type=image/jpeg"

# 5. containerize
docker build -t mlops-catsdogs:local .
docker run --rm -p 8000:8000 mlops-catsdogs:local

# 6. deploy (compose) + smoke test + monitor
IMAGE=mlops-catsdogs:local docker compose -f deploy/docker-compose.yml up -d
bash scripts/smoke_test.sh http://localhost:8000
python scripts/monitor_performance.py --base-url http://localhost:8000 --data-dir data/processed/test
```

A `Makefile` wraps every step: `make help`.

---

## Repository layout

```
.
├── src/mlops_catsdogs/          # importable package
│   ├── config.py                # paths, hyperparams, MLflow URI (env-overridable)
│   ├── data/
│   │   ├── preprocess.py        # resize 224x224 RGB, discover, split 80/10/10, manifest
│   │   ├── augment.py           # flip / rotate / colour-jitter + normalise -> tensor
│   │   └── dataset.py           # torch Dataset over processed folders
│   ├── model/
│   │   ├── cnn.py               # SimpleCNN baseline
│   │   └── infer.py             # softmax, load_model, predict_* , InferenceEngine
│   └── train.py                 # training loop + MLflow params/metrics/artifacts
├── app/
│   ├── main.py                  # FastAPI: /health /predict /metrics /metrics/summary
│   └── schemas.py               # pydantic request/response models
├── tests/                       # pytest: preprocessing, inference utils, API
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── smoke_test.sh            # post-deploy health + prediction check
│   ├── monitor_performance.py   # post-deployment accuracy / latency report
│   └── package_submission.sh    # build the deliverable zip
├── deploy/
│   ├── docker-compose.yml       # inference + Prometheus
│   ├── prometheus.yml
│   ├── k8s/                     # Deployment + Service + kustomization
│   └── argocd/application.yaml  # optional GitOps
├── .github/workflows/
│   ├── ci.yml                   # test -> build -> push image (GHCR)
│   └── cd.yml                   # pull image -> deploy -> smoke test
├── dvc.yaml / params.yaml       # reproducible pipeline (preprocess -> train)
├── Dockerfile / .dockerignore
├── requirements.txt             # pinned runtime deps
└── requirements-dev.txt
```

---

## M1 — Model development & experiment tracking

| Requirement | Where |
|---|---|
| Git source versioning | this repo |
| DVC dataset versioning | `dvc.yaml`, `params.yaml`, `data/raw.dvc`, `.dvc/` — `dvc repro` reproduces preprocess → train; `dvc add data/raw` tracks the raw set; a local remote is configured in `.dvc/config` (swap for S3/GDrive/SSH in production) |
| Baseline model | `SimpleCNN` (4 conv blocks + GAP + linear) in `src/mlops_catsdogs/model/cnn.py` |
| Serialized model | `models/model.pt` — `torch.save({state_dict, classes, arch, meta})` |
| Experiment tracking | **MLflow** (`sqlite:///mlflow.db` backend). `train.py` logs params (epochs, batch size, LR, seed, split sizes), per-epoch `train/val loss & accuracy`, final `test_{loss,accuracy,precision,recall,f1}`, and artifacts: `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `history.json`, and the model. |

Pre-processing: every image → `RGB`, resized to **224×224**, ImageNet-normalised;
deterministic **80 / 10 / 10** train/val/test split (`seed=42`); train-time
augmentation = random horizontal flip + ±15° rotation + brightness/contrast/
saturation jitter.

> **Note on the synthetic dataset.** `scripts/generate_synthetic_data.py` writes a
> small, easily separable stand-in set so the full pipeline runs without the
> ~800 MB Kaggle download (CI uses it too). On it the baseline trivially reaches
> ~1.0 accuracy. Point `data/raw/` at the real Kaggle dataset for representative
> numbers.

---

## M2 — Packaging & containerization

* **Inference service** — FastAPI (`app/main.py`):
  * `GET /health` → `{"status":"ok","model_loaded":true,...}`
  * `POST /predict` (multipart image) → `{"label","label_index","confidence","probabilities":{cat,dog},"inference_ms","request_id"}`
  * `GET /metrics` (Prometheus) and `GET /metrics/summary` (JSON counters) — see M5
* **Environment** — `requirements.txt` with **every key ML lib version-pinned**
  (`torch==2.9.0`, `numpy==1.26.4`, `pillow==10.2.0`, `scikit-learn==1.2.2`,
  `mlflow==3.14.0`, `fastapi==0.139.0`, …).
* **Containerization** — `Dockerfile` (python:3.11-slim, CPU-only torch wheel,
  non-root `appuser`, `HEALTHCHECK` on `/health`). Build & verify:

  ```bash
  docker build -t mlops-catsdogs:local .
  docker run --rm -p 8000:8000 mlops-catsdogs:local
  curl -s -X POST localhost:8000/predict -F "file=@data/processed/test/dog/$(ls data/processed/test/dog | head -1);type=image/jpeg"
  ```

---

## M3 — CI pipeline (`.github/workflows/ci.yml`)

On **every push / PR**:

1. `checkout` → `setup-python` → install `requirements-dev.txt` (+ CPU torch).
2. **Lint** — `ruff check .`
3. **Unit tests** — `pytest` with coverage. Tests cover:
   * `tests/test_preprocess.py` — `resize_to_square`, `image_to_array`,
     `label_from_path`, `split_dataset` (ratios, disjointness, determinism, bad input).
   * `tests/test_infer.py` — `softmax` (sums to 1, stable, monotonic),
     `load_model`, `predict_array` / `predict_bytes` output contract.
   * `tests/test_api.py` — `/health`, `/predict`, error handling, `/metrics`.
4. **Build** the Docker image (Buildx, GHA layer cache). On PRs it also boots the
   image and runs `scripts/smoke_test.sh`.
5. **Publish** — on pushes, log in to **GHCR** and push
   `ghcr.io/<owner>/<repo>` tagged with the commit SHA, branch, and `latest`
   (default branch). Swap the registry block for Docker Hub / a local registry if
   preferred.

---

## M4 — CD pipeline & deployment (`.github/workflows/cd.yml`)

Triggered by a **successful CI run on `main`** (`workflow_run`) or manually.

* **`deploy-compose`** (works out of the box on the runner): GHCR login → `docker
  pull` the new `latest` image → `docker compose -f deploy/docker-compose.yml up
  -d` → **post-deploy smoke test** (`scripts/smoke_test.sh`) that calls `/health`
  and one `/predict`; the job **fails** if either check fails.
* **`deploy-k8s`** (opt-in via repo var `ENABLE_K8S_DEPLOY=true` + `KUBE_CONFIG`
  secret): points `kubectl` at a kind/minikube/microk8s/VM cluster, sets the image
  tag via `kustomize`, `kubectl apply -f deploy/k8s`, waits for `rollout status`,
  then port-forwards and runs the same smoke test.
* **GitOps alternative** — `deploy/argocd/application.yaml`: Argo CD watches
  `deploy/k8s/` and auto-syncs (`prune` + `selfHeal`) when the CD job commits a new
  image tag.

Manifests: `deploy/k8s/deployment.yaml` (2 replicas, readiness/liveness on
`/health`, resource limits, rolling update) + `deploy/k8s/service.yaml`
(NodePort 30080). Compose: `deploy/docker-compose.yml`.

---

## M5 — Monitoring, logs & final submission

* **Request/response logging** — a FastAPI middleware logs one structured JSON
  line per request (`request_id`, method, path, status, `latency_ms`, client) and
  one per prediction (`label`, `confidence`, `inference_ms`). **Image bytes are
  never logged.**
* **Metrics**
  * `GET /metrics` — Prometheus exposition: `http_requests_total{method,path,status}`,
    `predictions_total{label}`, `prediction_errors_total`,
    `request_latency_seconds` histogram. Scraped by the Prometheus service in
    `deploy/docker-compose.yml` (`:9090`).
  * `GET /metrics/summary` — in-app counters: request count, prediction count,
    error count, avg & p95 latency (no external infra needed).
* **Post-deployment performance tracking** — `scripts/monitor_performance.py`
  replays a labelled batch (`data/processed/test/`) against the live service and
  writes `metrics/post_deploy_report.json`: accuracy, precision/recall/F1,
  confusion matrix, and latency percentiles.

---

## Reproducibility notes

* All randomness seeded (`SEED=42`): Python `random`, NumPy, PyTorch.
* Dependency versions pinned in `requirements*.txt`; container pins the Python
  minor version and installs the CPU torch wheel.
* `dvc repro` re-runs `preprocess → train` when inputs/params change.
* Config is environment-overridable (`EPOCHS`, `BATCH_SIZE`, `MODEL_PATH`,
  `MLFLOW_TRACKING_URI`, `RAW_DATA_DIR`, …) — see `src/mlops_catsdogs/config.py`.

## Deliverable

`bash scripts/package_submission.sh` produces `dist/mlops-catsdogs-submission.zip`
with all source, configs (DVC / CI-CD / Docker / manifests), the trained
`models/model.pt`, and the logged MLflow artifacts. See `docs/SUBMISSION.md` for
the marks-to-artifact map and the 5-minute screen-recording script.
