#!/usr/bin/env bash
# Post-deploy smoke test (M4 task 3).
# Calls the health endpoint and one prediction; exits non-zero on any failure so
# the CI/CD pipeline fails fast.
#
# Usage: scripts/smoke_test.sh [BASE_URL]
#   BASE_URL defaults to http://localhost:8000
set -euo pipefail

BASE_URL="${1:-${BASE_URL:-http://localhost:8000}}"
SAMPLE_IMAGE="${SAMPLE_IMAGE:-assets/sample_dog.jpg}"
RETRIES="${RETRIES:-30}"

echo "[smoke] target: ${BASE_URL}"

# 1. wait for readiness
for i in $(seq 1 "${RETRIES}"); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "[smoke] service is up (attempt ${i})"
    break
  fi
  if [ "${i}" -eq "${RETRIES}" ]; then
    echo "[smoke] FAIL: service not healthy after ${RETRIES} attempts"
    exit 1
  fi
  sleep 2
done

# 2. health assertion
HEALTH="$(curl -fsS "${BASE_URL}/health")"
echo "[smoke] /health -> ${HEALTH}"
echo "${HEALTH}" | grep -q '"status":"ok"' || { echo "[smoke] FAIL: bad health payload"; exit 1; }

# 3. prediction assertion
if [ ! -f "${SAMPLE_IMAGE}" ]; then
  echo "[smoke] generating a sample image"
  python - <<'PY'
import numpy as np, pathlib
from PIL import Image
pathlib.Path("assets").mkdir(exist_ok=True)
arr = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
Image.fromarray(arr, "RGB").save("assets/sample_dog.jpg")
PY
fi

PRED="$(curl -fsS -X POST "${BASE_URL}/predict" -F "file=@${SAMPLE_IMAGE};type=image/jpeg")"
echo "[smoke] /predict -> ${PRED}"
echo "${PRED}" | grep -Eq '"label":\s*"(cat|dog)"' || { echo "[smoke] FAIL: no label in prediction"; exit 1; }
echo "${PRED}" | grep -q '"probabilities"' || { echo "[smoke] FAIL: no probabilities in prediction"; exit 1; }

echo "[smoke] PASS"
