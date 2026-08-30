#!/usr/bin/env bash
# Build the deliverable zip:
#   - all source code + configuration files (DVC, CI/CD, Docker, deploy manifests)
#     -> taken straight from the git tree, so it is exactly what is committed
#   - the trained model artifact (models/model.pt is committed to git)
#   - MLflow experiment-tracking artifacts (loss/accuracy curves, confusion
#     matrix, history.json, test_metrics.json) from the latest local run
#
# The raw / processed datasets are intentionally NOT included: they are versioned
# with DVC and represented in the zip by dvc.yaml, dvc.lock, data/raw.dvc and
# .dvc/config.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="dist/mlops-catsdogs-submission.zip"
mkdir -p dist
rm -f "$OUT"

# 1. Refuse to package a dirty tree (the zip must match a real commit).
#    Override for a quick draft with:  ALLOW_DIRTY=1 scripts/package_submission.sh
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree has uncommitted changes:" >&2
  git status --short >&2
  if [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "Commit them first (or re-run with ALLOW_DIRTY=1)." >&2
    exit 1
  fi
  echo "ALLOW_DIRTY=1 set -> packaging committed content only; the changes above are NOT in the zip." >&2
fi

REV="$(git rev-parse --short HEAD)"
echo "packaging commit ${REV}"

# 2. All tracked files (source + every config + models/model.pt) via git archive.
git archive --format=zip --prefix="mlops-catsdogs/" -o "$OUT" HEAD

# 3. MLflow artifacts from the most recent local run -> plain folder, then append.
STAGE="$(mktemp -d)/mlops-catsdogs/mlartifacts_export"
mkdir -p "$STAGE"
if [ -d mlruns ]; then
  LATEST_RUN_DIR="$(find mlruns -maxdepth 4 -name artifacts -type d 2>/dev/null | sort | tail -1 || true)"
  [ -n "${LATEST_RUN_DIR:-}" ] && cp -R "${LATEST_RUN_DIR}/." "$STAGE/" 2>/dev/null || true
fi
cp -R metrics/. "$STAGE/" 2>/dev/null || true
[ -f mlflow.db ] && cp mlflow.db "$STAGE/" || true

( cd "$(dirname "$(dirname "$STAGE")")" && zip -rq "$OLDPWD/$OUT" "mlops-catsdogs/mlartifacts_export" )

# 4. Report.
SIZE="$(du -h "$OUT" | cut -f1)"
COUNT="$(unzip -l "$OUT" | tail -1 | awk '{print $2}')"
echo
echo "built $OUT  (${SIZE}, ${COUNT} files)"
echo "---- top-level contents ----"
unzip -l "$OUT" | awk '{print $4}' | grep -E '^mlops-catsdogs/[^/]+/?$' | sort -u
