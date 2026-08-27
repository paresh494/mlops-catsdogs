"""A tiny ``torch.utils.data.Dataset`` over the processed image folders."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from mlops_catsdogs.config import CLASS_NAMES
from mlops_catsdogs.data.preprocess import SUPPORTED_EXT, label_from_path


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path | str, split: str, transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for cls_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((p, cls_idx))
        if not self.samples:  # fall back to flat folder with cat./dog. filenames
            for p in sorted(self.root.rglob("*")):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((p, label_from_path(p)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.transform(im) if self.transform else im
        return x, label
