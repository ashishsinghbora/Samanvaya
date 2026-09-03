"""
Deep Space Transformer Matching Engine.
Utilizes a detector-free, coarse-to-fine visual transformer (LoFTR/RoMa architecture) 
to establish dense correspondences across highly divergent multi-modal lunar datasets.
"""

import os
from pathlib import Path
from typing import Tuple, Dict
import numpy as np

import torch
import torch.nn.functional as F
# Kornia provides production-ready, differentiable CV modules
from kornia.feature import LoFTR
from kornia.utils import image_to_tensor

class DeepSpaceTransformer:
    """
    Enterprise-grade wrapper for transformer-based image registration.
    Engineered to prevent CUDA OOM crashes during massive orbital swath processing.
    """

    def __init__(self, model_weights_path: str = "outdoor", device: str = None, 
                 confidence_threshold: float = 0.85):
        """
        Initializes the Transformer matching model.
        
        Args:
            model_weights_path: Path to local state_dict or Kornia preset (e.g., 'outdoor').
                                In an air-gapped ISRO environment, this must be a local Path.
            device: 'cuda', 'cpu', or 'mps'. Auto-detects if None.
            confidence_threshold: Minimum attention score to accept a correspondence.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.confidence_threshold = confidence_threshold
        
        # Load LoFTR. In production, load from a verified, hashed local .pt file
        self.matcher = LoFTR(pretrained=model_weights_path).to(self.device)
        self.matcher.eval()

    def _sanitize_and_pad(self, img: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy tile to a standardized PyTorch tensor.
        Transformers require image dimensions to be divisible by 8 or 16.
        """
        if img.ndim != 2:
            raise ValueError(f"Expected 2D grayscale array, got {img.ndim}D.")

        # Convert to tensor, add batch and channel dims [B, C, H, W], normalize to [0, 1]
        tensor = image_to_tensor(img, keepdim=False).float() / 255.0
        tensor = tensor.to(self.device)
        
        # Calculate padding to ensure divisibility by 8 (LoFTR requirement)
        _, _, h, w = tensor.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        
        if pad_h > 0 or pad_w > 0:
            # Pad bottom and right edges using reflection to avoid hard edge artifacts
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
            
        return tensor

    @torch.inference_mode()
    def match_tiles(self, src_tile: np.ndarray, ref_tile: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Executes the transformer forward pass to extract dense tie points.
        Uses Automatic Mixed Precision (AMP) to halve GPU VRAM footprint.
        
        Args:
            src_tile: 2D numpy array of the Chandrayaan-2 moving image.
            ref_tile: 2D numpy array of the LRO NAC fixed image.
            
        Returns:
            Dictionary containing 'src_pts', 'ref_pts', and 'confidence' arrays.
        """
        # 1. Sanitize, pad, and upload to device
        t_src = self._sanitize_and_pad(src_tile)
        t_ref = self._sanitize_and_pad(ref_tile)
        
        # 2. Prevent VRAM exhaustion on oversized tiles
        max_dim = 1200
        if t_src.shape[2] > max_dim or t_src.shape[3] > max_dim:
            raise RuntimeError(f"Tile exceeds ML safety bounds {max_dim}x{max_dim}. Use smaller chunking.")
            
        # 3. Inference with Automatic Mixed Precision (FP16)
        input_dict = {"image0": t_src, "image1": t_ref}
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            correspondences = self.matcher(input_dict)
            
        # 4. Filter by confidence and move back to CPU
        mask = correspondences['confidence'] > self.confidence_threshold
        
        src_kpts = correspondences['keypoints0'][mask].cpu().numpy()
        ref_kpts = correspondences['keypoints1'][mask].cpu().numpy()
        conf = correspondences['confidence'][mask].cpu().numpy()
        
        return {
            "src_pts": src_kpts,
            "ref_pts": ref_kpts,
            "confidence": conf
        }
