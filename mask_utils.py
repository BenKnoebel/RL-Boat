"""
Utilities for building binary masks from images.

The primary helper here turns bright/white regions into 1s and darker/gray
regions into 0s, which matches the needs of the provided `image.png`.
"""

from __future__ import annotations

import numpy as np
from matplotlib import image as mpimg
from matplotlib import pyplot as plt


def _otsu_threshold(values: np.ndarray) -> float:
    """Compute an Otsu threshold for values normalized to [0, 1]."""
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    # Between-class variance; small epsilon avoids division by zero.
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return float(bin_centers[np.argmax(sigma_b2)])


def create_white_mask(
    image_path: str,
    output_path: str | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Create a binary mask where white/bright areas are 1 and gray/dark are 0.

    Args:
        image_path: Path to the input image (e.g., `image.png`).
        output_path: Optional path to save the mask as a PNG.
        threshold: Optional manual threshold in [0, 1]. If omitted, Otsu is used.

    Returns:
        NumPy array mask of dtype uint8 with values 0 or 1.
    """
    img = mpimg.imread(image_path)

    # Normalize to [0, 1] if the image is in 0-255 integer format.
    if img.dtype.kind in ("u", "i"):
        img = img.astype(np.float32) / 255.0

    # Drop alpha channel if present.
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    # Convert to grayscale luminance.
    if img.ndim == 3:
        # Standard Rec. 601 luma weights.
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = (img[..., :3] * weights).sum(axis=-1)
    else:
        gray = img.astype(np.float32)

    if threshold is None:
        threshold = _otsu_threshold(gray.ravel())

    mask = (gray >= threshold).astype(np.uint8)

    if output_path is not None:
        plt.imsave(output_path, mask, cmap="gray", vmin=0, vmax=1)

    return mask
