# Submission guide — S1-25 AIMLCZG523 Assignment 2

## 1. Deliverable zip

```bash
bash scripts/package_submission.sh
# -> dist/mlops-catsdogs-submission.zip
```

Contents:

| Deliverable item | Included |
|---|---|
| All source code | `src/`, `app/`, `tests/`, `scripts/` |
| DVC config | `dvc.yaml`, `params.yaml`, `data/raw.dvc`, `.dvc/config`, `.dvcignore` |
| CI/CD config | `.github/workflows/ci.yml`, `.github/workflows/cd.yml` |
| Docker config | `Dockerfile`, `.dockerignore`, `deploy/docker-compose.yml`, `deploy/prometheus.yml` |
| Deployment manifests | `deploy/k8s/*.yaml`, `deploy/argocd/application.yaml` |
| Trained model artifact | `models/model.pt` |
| Experiment-tracking artifacts | `mlartifacts_export/` (loss curve, accuracy curve, confusion matrix, `history.json`, `test_metrics.json`) + `metrics/` |

## 2. Marks → artifact map

### M1 — Model Development & Experiment Tracking (10)
- **Data & code versioning** — Git history; `dvc.yaml` (2-stage pipeline), `data/raw.dvc`, `params.yaml`.
- **Model building** — `src/mlops_catsdogs/model/cnn.py` (`SimpleCNN`); serialized to `models/model.pt`. The checkpoint is committed to Git (not a DVC output) so it is baked into the Docker image for real inference; the raw dataset and `data/processed` remain DVC-tracked.
- **Experiment tracking** — `src/mlops_catsdogs/train.py` logs params, per-epoch metrics, test metrics, and artifacts to MLflow. Show `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### M2 — Packaging & Containerization (10)
- **Inference service** — `app/main.py`; endpoints `GET /health`, `POST /predict` (+ `/metrics`).
- **Environment spec** — `requirements.txt`, every key ML lib pinned.
- **Containerization** — `Dockerfile`; `docker build` + `docker run` + `curl /predict`.

### M3 — CI Pipeline (10)
- **Automated testing** — `tests/test_preprocess.py` (pre-processing fns) + `tests/test_infer.py` (inference utils) + `tests/test_api.py`; run with `pytest`.
- **CI setup** — `.github/workflows/ci.yml` (GitHub Actions): checkout → install → `ruff` → `pytest` → `docker build`.
- **Artifact publishing** — same workflow pushes the image to **GHCR** (`ghcr.io/<owner>/<repo>:<sha>` + `latest`).

### M4 — CD Pipeline & Deployment (10)
- **Deployment target** — `deploy/docker-compose.yml` (primary) and `deploy/k8s/` Deployment+Service (K8s option).
- **CD / GitOps** — `.github/workflows/cd.yml` pulls the new image and redeploys on `main`; `deploy/argocd/application.yaml` for Argo CD.
- **Smoke tests** — `scripts/smoke_test.sh` runs post-deploy and fails the pipeline on any failure.

### M5 — Monitoring, Logs & Final Submission (10)
- **Logging & metrics** — request/response logging middleware in `app/main.py`; `/metrics` (Prometheus) + `/metrics/summary`; Prometheus service in compose.
- **Post-deployment performance** — `scripts/monitor_performance.py` → `metrics/post_deploy_report.json`.
- **Final submission** — this zip + screen recording.

## 3. Screen recording (< 5 min) — suggested script

1. **(0:00)** Show the repo + `git log`. Open `mlflow ui`, show a run's params, metric curves, confusion matrix, and the logged `model.pt`.
2. **(0:45)** Make a small code change (e.g. bump `params.yaml` `epochs`, or edit a preprocessing constant). `git commit && git push` to a branch / open a PR.
3. **(1:30)** GitHub → Actions: CI runs — lint, `pytest` green, Docker image built; on merge to `main`, image pushed to GHCR (show the package).
4. **(2:45)** CD workflow triggers: pulls the new image, `docker compose up`, runs `scripts/smoke_test.sh` — health + prediction pass.
5. **(3:30)** `curl` the deployed `/health` and `/predict` with a cat and a dog image; show the JSON labels/probabilities.
6. **(4:00)** Show `/metrics` (Prometheus) or Prometheus at `:9090`; run `scripts/monitor_performance.py` and open `metrics/post_deploy_report.json` (accuracy + latency).
7. **(4:45)** Recap the pipeline diagram in `README.md`.

## 4. Switching to the real Kaggle dataset

```bash
# download "Dogs vs Cats" from Kaggle, then:
rm -rf data/raw/* && unzip ~/Downloads/dogs-vs-cats.zip -d data/raw
#   expected: data/raw/train/cat.0.jpg ... dog.0.jpg   (or cats/ & dogs/ folders)
dvc add data/raw && git add data/raw.dvc data/.gitignore
python -m mlops_catsdogs.data.preprocess
python -m mlops_catsdogs.train --epochs 15 --batch-size 64
```
