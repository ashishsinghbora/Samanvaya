"""
Sub-Pixel Refinement Engine.
Uses 2D paraboloid fitting on Normalized Cross-Correlation (NCC) surfaces
to achieve < 0.1 pixel accuracy for lunar image correspondences.
"""

import numpy as np
import cv2
from typing import Tuple

class SubPixelRefiner:
    def __init__(self, patch_size: int = 31):
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

    def refine_match(self, src_img: np.ndarray, ref_img: np.ndarray, 
                     src_pt: Tuple[float, float], ref_pt: Tuple[float, float]) -> Tuple[float, float]:
        """
        Refines an integer coordinate match to sub-pixel accuracy.
        """
        sx, sy = int(round(src_pt[0])), int(round(src_pt[1]))
        rx, ry = int(round(ref_pt[0])), int(round(ref_pt[1]))

        # Extract small template from src and slightly larger search window from ref
        src_patch = self._get_patch(src_img, sx, sy, radius=self.half_patch // 2)
        ref_patch = self._get_patch(ref_img, rx, ry, radius=self.half_patch)

        if src_patch is None or ref_patch is None:
            return ref_pt # Fallback to original if out of bounds

        # Compute cross-correlation surface
        corr_surface = cv2.matchTemplate(ref_patch, src_patch, cv2.TM_CCORR_NORMED)
        
        # Find integer peak in the correlation surface
        _, _, _, max_loc = cv2.minMaxLoc(corr_surface)
        px, py = max_loc
        
        # Center of correlation surface corresponds to zero relative displacement
        center_x = (corr_surface.shape[1] - 1) / 2.0
        center_y = (corr_surface.shape[0] - 1) / 2.0

        # Fit 2D parabola around the peak to find sub-pixel maximum
        dx, dy = self._fit_2d_parabola(corr_surface, px, py)
        
        # Calculate final sub-pixel coordinate in the reference image
        refined_rx = rx + (px - center_x) + dx
        refined_ry = ry + (py - center_y) + dy

        return (float(refined_rx), float(refined_ry))

    def _get_patch(self, img: np.ndarray, x: int, y: int, radius: int = None) -> np.ndarray:
        """Safely extracts a patch with boundary checking."""
        r = radius if radius is not None else self.half_patch
        h, w = img.shape
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return None
        return img[y-r : y+r+1, x-r : x+r+1]

    def _fit_2d_parabola(self, surface: np.ndarray, px: int, py: int) -> Tuple[float, float]:
        """Solves the Taylor expansion for the sub-pixel peak shift."""
        if px == 0 or px == surface.shape[1]-1 or py == 0 or py == surface.shape[0]-1:
            return 0.0, 0.0 # Cannot interpolate on edges

        # Neighboring correlation values
        c = surface[py, px]
        cx1 = surface[py, px-1]
        cx2 = surface[py, px+1]
        cy1 = surface[py-1, px]
        cy2 = surface[py+1, px]

        # First and second derivatives
        dx = (cx2 - cx1) / 2.0
        dy = (cy2 - cy1) / 2.0
        dxx = cx1 - 2*c + cx2
        dyy = cy1 - 2*c + cy2

        # Sub-pixel shift
        shift_x = -dx / dxx if dxx != 0 else 0.0
        shift_y = -dy / dyy if dyy != 0 else 0.0

        # Bound the shift to [-1.0, 1.0] to prevent unstable fits
        return np.clip(shift_x, -1.0, 1.0), np.clip(shift_y, -1.0, 1.0)
