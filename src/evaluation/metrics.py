"""
src/evaluation/metrics.py

Defense-grade photogrammetric evaluation metrics.
Verifies compliance with ISRO's < 0.40 px RMSE and spatial distribution mandates.
"""
import numpy as np

class EvaluationEngine:
    @staticmethod
    def compute_rmse(pts1: np.ndarray, pts2: np.ndarray) -> float:
        """
        Computes Root Mean Squared Error (RMSE) between corresponding point pairs.
        Returns sub-pixel accuracy metric.
        """
        if len(pts1) == 0 or len(pts2) == 0:
            return float('inf')
            
        diff = pts1 - pts2
        sq_dist = np.sum(diff**2, axis=1)
        rmse = np.sqrt(np.mean(sq_dist))
        return float(rmse)

    @staticmethod
    def compute_spatial_entropy(pts: np.ndarray, image_shape: tuple, grid_size: int = 8) -> float:
        """
        Computes Normalized Shannon Entropy to verify spatial uniformity.
        H >= 0.95 indicates excellent, uniform distribution across the image.
        """
        if len(pts) == 0:
            return 0.0

        h, w = image_shape[:2]
        bins_x = np.linspace(0, w, grid_size + 1)
        bins_y = np.linspace(0, h, grid_size + 1)

        # 2D Histogram counts points in each grid cell
        counts, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[bins_x, bins_y])
        
        # Flatten and compute probability distribution
        p = counts.flatten() / np.sum(counts)
        p = p[p > 0] # Avoid log2(0)

        entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(grid_size * grid_size)
        
        normalized_entropy = entropy / max_entropy
        return float(normalized_entropy)
