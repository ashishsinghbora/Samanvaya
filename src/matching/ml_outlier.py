"""
Advanced Geometric Outlier Rejection.
Uses MAGSAC++ (Marginalizing Sample Consensus) to robustly compute the affine/homography 
matrices from noisy deep-learning matches.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class RobustEstimator:
    @staticmethod
    def filter_matches(src_pts: np.ndarray, ref_pts: np.ndarray, 
                       confidence_weights: np.ndarray, 
                       error_threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Fundamental Matrix using MAGSAC++ and returns only the strictly verified inliers.
        """
        if len(src_pts) < 8:
            return np.empty((0,2)), np.empty((0,2))
            
        # MAGSAC++ utilizes continuous noise modeling rather than strict inlier thresholds, 
        # making it vastly superior to standard RANSAC for ML-generated tie points.
        F, inlier_mask = cv2.findFundamentalMat(
            src_pts, 
            ref_pts, 
            cv2.USAC_MAGSAC, 
            error_threshold, 
            0.999, # 99.9% confidence required
            100_000 # Max iterations
        )
        
        if inlier_mask is None:
            return np.empty((0,2)), np.empty((0,2))
            
        mask_bool = inlier_mask.ravel() == 1
        
        return src_pts[mask_bool], ref_pts[mask_bool]
