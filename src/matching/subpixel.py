"""
Sub-Pixel Refinement Engine (Antigravity JIT Edition).
Uses Numba Just-In-Time compilation to achieve weightless, near-instantaneous 
execution of 2D paraboloid fitting for < 0.1 pixel accuracy.
"""

import numpy as np
import cv2
try:
    from numba import njit
except ImportError:
    # Fallback decorator if numba is not installed in the environment
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from typing import Tuple

@njit(fastmath=True)
def _fit_2d_parabola_jit(surface: np.ndarray, px: int, py: int) -> Tuple[float, float]:
    """
    JIT-compiled Taylor expansion for sub-pixel peak shift.
    Bypasses the Python interpreter for C-level execution speed.
    """
    # Cannot interpolate on the absolute edges
    if px == 0 or px == surface.shape[1]-1 or py == 0 or py == surface.shape[0]-1:
        return 0.0, 0.0 

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
    shift_x = -dx / dxx if dxx != 0.0 else 0.0
    shift_y = -dy / dyy if dyy != 0.0 else 0.0

    # Bound the shift to [-1.0, 1.0] to prevent unstable fits
    # Using manual min/max for strict Numba compatibility
    shift_x = max(-1.0, min(shift_x, 1.0))
    shift_y = max(-1.0, min(shift_y, 1.0))

    return shift_x, shift_y


class SubPixelRefiner:
    def __init__(self, patch_size: int = 31):
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

    def refine_match(self, src_img: np.ndarray, ref_img: np.ndarray, 
                     src_pt: Tuple[float, float], ref_pt: Tuple[float, float]) -> Tuple[float, float]:
        """Refines an integer coordinate match to sub-pixel accuracy."""
        sx, sy = int(round(src_pt[0])), int(round(src_pt[1]))
        rx, ry = int(round(ref_pt[0])), int(round(ref_pt[1]))

        # Extract small template from src and search window from ref
        src_patch = self._get_patch(src_img, sx, sy, radius=self.half_patch // 2)
        ref_patch = self._get_patch(ref_img, rx, ry, radius=self.half_patch)

        if src_patch is None or ref_patch is None:
            return float(ref_pt[0]), float(ref_pt[1])

        # Compute cross-correlation surface
        corr_surface = cv2.matchTemplate(ref_patch, src_patch, cv2.TM_CCORR_NORMED)
        
        # Find integer peak
        _, _, _, max_loc = cv2.minMaxLoc(corr_surface)
        px, py = max_loc

        # Center of correlation surface corresponds to zero relative displacement
        center_x = (corr_surface.shape[1] - 1) / 2.0
        center_y = (corr_surface.shape[0] - 1) / 2.0

        # Defy gravity: Execute JIT-compiled math
        dx, dy = _fit_2d_parabola_jit(corr_surface, px, py)
        
        refined_rx = rx + (px - center_x) + dx
        refined_ry = ry + (py - center_y) + dy

        return (float(refined_rx), float(refined_ry))

    def _get_patch(self, img: np.ndarray, x: int, y: int, radius: int = None):
        """Safely extracts a patch with boundary checking."""
        r = radius if radius is not None else self.half_patch
        h, w = img.shape
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return None
        return img[y-r : y+r+1, x-r : x+r+1]
