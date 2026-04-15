"""Classical perception: segment the gray track and extract a centerline.

Pipeline:
  frame (HxWx3) --HSV threshold--> binary track mask
                --morphological close--> cleaned mask
                --row-wise centroid--> centerline points [(row, col), ...]
"""
import cv2
import numpy as np


# Track = gray asphalt + yellow centerline stripe. Grass is green (high sat).
# Asphalt: any hue, saturation < ~50. Yellow stripe: hue ~25-35, high sat.
GRAY_LOW = np.array([0, 0, 40], dtype=np.uint8)
GRAY_HIGH = np.array([180, 60, 200], dtype=np.uint8)
YELLOW_LOW = np.array([15, 80, 80], dtype=np.uint8)
YELLOW_HIGH = np.array([40, 255, 255], dtype=np.uint8)


def track_mask(frame: np.ndarray) -> np.ndarray:
    """Return a binary mask (uint8, 0 or 255) of the track pixels (asphalt + yellow stripe)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    gray = cv2.inRange(hsv, GRAY_LOW, GRAY_HIGH)
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    mask = cv2.bitwise_or(gray, yellow)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def centerline(mask: np.ndarray) -> list[tuple[int, int]]:
    """For each row, take the centroid column of track pixels. Returns [(row, col)]."""
    points = []
    for r in range(mask.shape[0]):
        cols = np.where(mask[r] > 0)[0]
        if cols.size > 0:
            points.append((r, int(cols.mean())))
    return points
