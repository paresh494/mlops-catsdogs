"""Train the baseline SimpleCNN and log everything to MLflow.

Run:
    python -m mlops_catsdogs.train --epochs 5 --batch-size 32

Logs to MLflow: parameters, per-epoch train/val loss & accuracy, final test
metrics, and artifacts (loss curve, accuracy curve, confusion matrix, the
serialized model ``model.pt``).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlops_catsdogs.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PATHS,
    TRAIN,
)
from mlops_catsdogs.data.augment import build_transform
from mlops_catsdogs.data.dataset import ImageFolderDataset
from mlops_catsdogs.model.cnn import build_model
from mlops_catsdogs.model.infer import save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(split: str, train: bool, batch_size: int, num_workers: int) -> DataLoader:
    ds = ImageFolderDataset(PATHS.processed_data, split, transform=build_transform(train=train))
    if len(ds) == 0:
        raise FileNotFoundError(
            f"no processed images for split '{split}' under {PATHS.processed_data}. "
            f"Run: python -m mlops_catsdogs.data.preprocess"
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, drop_last=False)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses, y_true, y_pred = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        y_pred += logits.argmax(1).cpu().tolist()
        y_true += y.cpu().tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc, y_true, y_pred


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = 2) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _line_plot(out_path: Path, train_series, val_series, ylabel: str, title: str, plt) -> Path:
    fig, ax = plt.subplots()
    ax.plot(train_series, label="train")
    ax.plot(val_series, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _confusion_plot(out_path: Path, cm: np.ndarray, plt) -> Path:
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["cat", "dog"])
    ax.set_yticklabels(["cat", "dog"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_plots(history: dict, cm: np.ndarray, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return []

    return [
        _line_plot(out_dir / "loss_curve.png", history["train_loss"], history["val_loss"], "loss", "Loss", plt),
        _line_plot(out_dir / "accuracy_curve.png", history["train_acc"], history["val_acc"], "accuracy", "Accuracy", plt),
        _confusion_plot(out_dir / "confusion_matrix.png", cm, plt),
    ]


def train(args: argparse.Namespace) -> dict:
    import mlflow

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = _loader("train", True, args.batch_size, args.num_workers)
    val_loader = _loader("val", False, args.batch_size, args.num_workers)
    test_loader = _loader("test", False, args.batch_size, args.num_workers)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    artifacts_dir = PATHS.metrics
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "optimizer": "Adam",
                "arch": "SimpleCNN",
                "image_size": 224,
                "n_train": len(train_loader.dataset),
                "n_val": len(val_loader.dataset),
                "n_test": len(test_loader.dataset),
            }
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            ep_losses, y_true, y_pred = [], [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())
                y_pred += logits.argmax(1).detach().cpu().tolist()
                y_true += y.cpu().tolist()

            tr_loss = float(np.mean(ep_losses))
            tr_acc = float((np.array(y_true) == np.array(y_pred)).mean())
            va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            mlflow.log_metrics(
                {"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc},
                step=epoch,
            )
            print(f"epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  val_loss={va_loss:.4f} acc={va_acc:.3f}")

        # final test evaluation
        te_loss, te_acc, yt, yp = evaluate(model, test_loader, criterion, device)
        cm = _confusion_matrix(yt, yp)
        tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        test_metrics = {
            "test_loss": te_loss,
            "test_accuracy": te_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        }
        mlflow.log_metrics(test_metrics)
        print("test:", json.dumps(test_metrics, indent=2))

        # artifacts: plots, history json, model
        (artifacts_dir / "history.json").write_text(json.dumps(history, indent=2))
        (artifacts_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        (artifacts_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist()))
        for p in _save_plots(history, cm, artifacts_dir):
            mlflow.log_artifact(str(p), artifact_path="plots")
        mlflow.log_artifact(str(artifacts_dir / "history.json"))
        mlflow.log_artifact(str(artifacts_dir / "test_metrics.json"))

        model_path = save_checkpoint(
            model, PATHS.model_file, extra={"run_id": run.info.run_id, **test_metrics}
        )
        mlflow.log_artifact(str(model_path), artifact_path="model")
        print(f"saved model -> {model_path}")

        summary = {"run_id": run.info.run_id, "model_path": str(model_path), **test_metrics}
        (artifacts_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train SimpleCNN on Cats vs Dogs")
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    ap.add_argument("--learning-rate", type=float, default=TRAIN.learning_rate)
    ap.add_argument("--weight-decay", type=float, default=TRAIN.weight_decay)
    ap.add_argument("--seed", type=int, default=TRAIN.seed)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
