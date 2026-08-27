"""Post-deployment model performance tracking (M5 task 2).

Sends a batch of labelled images to the deployed service, collects predictions,
and computes accuracy / precision / recall / confusion matrix plus latency
statistics. Writes a JSON report and prints a summary.

Usage:
    python scripts/monitor_performance.py \
        --base-url http://localhost:8000 \
        --data-dir data/processed/test \
        --limit 100 \
        --out metrics/post_deploy_report.json

``--data-dir`` is expected to contain ``cat/`` and ``dog/`` sub-folders (the
layout produced by the preprocessing stage).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

CLASSES = ["cat", "dog"]


def collect_samples(data_dir: Path, limit: int) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for idx, cls in enumerate(CLASSES):
        for p in sorted((data_dir / cls).glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((p, idx))
    samples.sort(key=lambda t: t[0].name)
    if limit:
        # interleave classes so a truncated run still covers both
        by_cls = {0: [], 1: []}
        for p, y in samples:
            by_cls[y].append((p, y))
        merged = []
        for a, b in zip(by_cls[0], by_cls[1]):
            merged += [a, b]
        samples = merged[:limit]
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed/test"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("metrics/post_deploy_report.json"))
    args = ap.parse_args()

    samples = collect_samples(args.data_dir, args.limit)
    if not samples:
        raise SystemExit(f"no labelled images found under {args.data_dir}")

    cm = [[0, 0], [0, 0]]  # cm[true][pred]
    latencies: list[float] = []
    n_ok = 0
    rows = []

    for path, y_true in samples:
        with open(path, "rb") as fh:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{args.base_url}/predict",
                files={"file": (path.name, fh, "image/jpeg")},
                timeout=30,
            )
            dt = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        body = resp.json()
        y_pred = int(body["label_index"])
        cm[y_true][y_pred] += 1
        latencies.append(dt)
        n_ok += int(y_pred == y_true)
        rows.append({"file": path.name, "true": CLASSES[y_true], "pred": body["label"],
                     "confidence": body["confidence"], "latency_ms": round(dt, 1)})

    n = len(samples)
    tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    latencies.sort()

    report = {
        "base_url": args.base_url,
        "n_samples": n,
        "accuracy": round(n_ok / n, 4),
        "precision_dog": round(precision, 4),
        "recall_dog": round(recall, 4),
        "f1_dog": round(f1, 4),
        "confusion_matrix": {"labels": CLASSES, "matrix": cm},
        "latency_ms": {
            "avg": round(sum(latencies) / n, 1),
            "p50": round(latencies[int(n * 0.50) - 1], 1),
            "p95": round(latencies[int(n * 0.95) - 1], 1),
            "max": round(latencies[-1], 1),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "samples": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    summary = {k: report[k] for k in ("n_samples", "accuracy", "precision_dog", "recall_dog", "f1_dog", "latency_ms")}
    print(json.dumps(summary, indent=2))
    print(f"full report -> {args.out}")


if __name__ == "__main__":
    main()
