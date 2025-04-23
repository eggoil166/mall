import cv2
import numpy as np

def compute_homography(og_pts, target_pts):
    """Compute homography matrix (paper Eq. 4)."""
    return cv2.findHomography(og_pts, target_pts)[0]

def transform_points(points, H):
    """Apply homography to points (paper Eq. 4)."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).squeeze()