"""
src/features/shadow_mask.py

Illumination invariance module.
Detects pitch-black solar shadows cast by crater rims to prevent false 
feature matching inside regions with zero information.
"""
import cv2
import numpy as np

class ShadowMasker:
    """
    Applies Otsu's thresholding and active contours to isolate lunar shadows.
    """
    
    @staticmethod
    def generate_mask(image: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask where 0 indicates a shadow, and 1 indicates valid terrain.
        """
        # Normalize to 0-255 uint8 if not already
        if image.dtype != np.uint8:
            norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            norm_img = image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(norm_img, (5, 5), 0)

        # Otsu's thresholding to find optimal shadow threshold
        thresh_val, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological opening to remove small isolated shadow pixels (noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Convert to 0 (shadow) and 1 (valid)
        binary_mask = (mask_opened > 0).astype(np.uint8)
        return binary_mask
