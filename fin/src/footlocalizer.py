import cv2
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import joblib
import os

class FootLocalizer:
    def __init__(self, background, gp_threshold=4e-4):
        self.background = background
        self.gp_threshold = gp_threshold
        self.gp_model = None
        self.head_foot_pairs = []

    @classmethod
    def build_background(cls, frames):
        return np.mean(frames, axis=0).astype(np.uint8)

    def add_training_pair(self, head_coord, foot_y):
        """Add (x, y_head) -> y_foot difference sample."""
        x, y = head_coord
        self.head_foot_pairs.append((np.array([x, y]), foot_y - y))

    def fit_gp_model(self):
        """Fit Gaussian Process model on collected head-foot pairs."""
        if not self.head_foot_pairs:
            return
        X, y = zip(*self.head_foot_pairs)
        X = np.vstack(X)
        y = np.array(y)
        kernel = RBF(length_scale=50)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1.0)
        self.gp_model.fit(X, y)

    def _compute_binary_mask(self, frame, threshold=70):
        diff = cv2.absdiff(frame, self.background)
        mask = (np.linalg.norm(diff, axis=2) > threshold).astype(np.uint8)
        return mask

    def _estimate_density(self, mask, num_people):
        return num_people / (np.sum(mask) + 1e-6)

    def get_mask_localize_init(self, mask, bbox):
        x1, y1, x2, y2 = map(int, np.array(bbox).flatten())
        sliced_map = mask[y2:, x1:x2]
        first_false = np.argmax(sliced_map == False, axis=1)
        has_false = np.any(sliced_map == False, axis=1)
        valid_rows = np.where(has_false)[0]
        final_edge = valid_rows + first_false[valid_rows] + y2
        x_center = (x1 + x2) / 2
        y_feet = np.mean(final_edge)
        return np.array([x_center, y_feet])
    
    def save_gp_model(self, path):
        if self.gp_model is not None:
            print("saved")
            joblib.dump(self.gp_model, path)
        else:
            print("no gp model")

    def load_gp_model(self, path):
        self.gp_model = joblib.load(path)