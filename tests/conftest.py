import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(140, 90, 3), dtype=np.uint8)  # non-square on purpose
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def png_bytes(rgb_image) -> bytes:
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    return buf.getvalue()
