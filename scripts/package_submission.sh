#!/usr/bin/env bash
# Build the deliverable zip: source + configs + trained model + MLflow artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="dist/mlops-catsdogs-submission.zip"
mkdir -p dist

# Export the latest MLflow run's artifacts into a plain folder for the zip.
rm -rf mlartifacts_export && mkdir -p mlartifacts_export
if [ -d mlruns ]; then
  LATEST_RUN_DIR="$(find mlruns -maxdepth 3 -name artifacts -type d | sort | tail -1 || true)"
  if [ -n "${LATEST_RUN_DIR:-}" ]; then
    cp -R "${LATEST_RUN_DIR}/." mlartifacts_export/ 2>/dev/null || true
  fi
fi
cp -R metrics/. mlartifacts_export/ 2>/dev/null || true

zip -r "$OUT" \
  src app tests scripts deploy .github \
  Dockerfile .dockerignore docker-compose.yml 2>/dev/null || true

# The line above tolerates missing optional files; now add the rest explicitly.
zip -r "$OUT" \
  README.md docs pyproject.toml pytest.ini Makefile \
  requirements.txt requirements-dev.txt \
  dvc.yaml params.yaml .dvcignore .dvc/config \
  data/raw.dvc data/.gitignore data/processed \
  models/model.pt \
  mlartifacts_export \
  -x '*/__pycache__/*' -x '*.pyc' -x '*/.pytest_cache/*' -x '*/.ruff_cache/*'

echo
echo "built $OUT"
unzip -l "$OUT" | tail -5
