from __future__ import annotations

import numpy as np


def create_detection_mask(
    snr_map: np.ndarray,
    threshold: float = 5.0,
) -> np.ndarray:
    """
    Create a boolean mask containing pixels above the SNR threshold.
    """
    if threshold <= 0:
        raise ValueError("Threshold must be greater than zero.")

    return np.isfinite(snr_map) & (snr_map >= threshold)

def label_sources(
    detection_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Label connected regions in a detection mask.

    Each connected region receives a unique positive integer label.
    Background pixels are labeled 0.

    Returns
    -------
    labels : np.ndarray
        Integer array with one label per detected source.
    count : int
        Number of detected sources.
    """
    if detection_mask.dtype != np.bool_:
        raise ValueError("Detection mask must be boolean.")

    height, width = detection_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)

    current_label = 0

    for y in range(height):
        for x in range(width):
            if not detection_mask[y, x] or labels[y, x] != 0:
                continue

            current_label += 1
            stack = [(y, x)]
            labels[y, x] = current_label

            while stack:
                cy, cx = stack.pop()

                for dy, dx in (
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1),
                ):
                    ny = cy + dy
                    nx = cx + dx

                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and detection_mask[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = current_label
                        stack.append((ny, nx))

    return labels, current_label