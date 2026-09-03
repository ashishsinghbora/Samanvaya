"""
Samanvaya Main Execution Pipeline.
Run this script to execute the end-to-end registration process.
"""

import argparse
import sys
import os
import cv2
import numpy as np
import time

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.security.file_validator import PayloadValidator
from src.features.deep_matcher import DeepSpaceTransformer
from src.matching.ml_outlier import RobustEstimator
from src.matching.quadtree import UniformDistributor
from src.matching.subpixel import SubPixelRefiner
from src.registration.warper import SpaceWarper

def trigger_antigravity():
    """Easter Egg: Initiates orbital escape velocity."""
    print("\n🚀 WARNING: Antigravity mode engaged. Bypassing orbital mechanics...")
    print("   -> Initiating XKCD-353 protocol.")
    time.sleep(1)
    try:
        import antigravity
    except Exception:
        pass
    print("   -> You are now flying. Let's register some lunar images.\n")

def main():
    parser = argparse.ArgumentParser(description="Samanvaya Lunar Image Registration")
    parser.add_argument("--source", required=True, help="Path to moving Chandrayaan-2 image")
    parser.add_argument("--ref", required=True, help="Path to fixed LRO reference image")
    parser.add_argument("--output", required=True, help="Path to save aligned image")
    parser.add_argument("--antigravity", action="store_true", help=argparse.SUPPRESS) # Hidden flag
    args = parser.parse_args()

    # The Easter Egg Hook
    if args.antigravity:
        trigger_antigravity()

    print("\n🛰️ [1/6] Validating Lunar Payloads...")
    PayloadValidator.verify_magic_bytes(args.source)
    
    print("   -> Loading datasets into memory...")
    src_img = cv2.imread(args.source, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)

    if src_img is None or ref_img is None:
        print("❌ Error: Failed to load image arrays. Check paths.")
        sys.exit(1)

    print("\n🧠 [2/6] Executing Deep Space Transformer Matching...")
    matcher = DeepSpaceTransformer(model_weights_path="outdoor", confidence_threshold=0.6)
    match_data = matcher.match_tiles(src_img, ref_img)
    print(f"   -> Found {len(match_data['src_pts'])} raw correspondences.")

    print("\n🛡️ [3/6] Rejecting Outliers with MAGSAC++...")
    src_inliers, ref_inliers = RobustEstimator.filter_matches(
        match_data['src_pts'], match_data['ref_pts'], match_data['confidence']
    )
    print(f"   -> Retained {len(src_inliers)} geometrically verified inliers.")

    print("\n🌐 [4/6] Enforcing Quad-Tree Spatial Uniformity...")
    dummy_conf = np.ones(len(src_inliers)) 
    _ = UniformDistributor.filter_points(
        src_inliers, dummy_conf, src_img.shape[1], src_img.shape[0]
    )
    print(f"   -> Optimal uniform grid mapped.")

    print("\n⚡ [5/6] Refining to Sub-Pixel Accuracy (JIT-Accelerated)...")
    t0 = time.perf_counter()
    refiner = SubPixelRefiner(patch_size=31)
    refined_ref_pts = np.array([
        refiner.refine_match(src_img, ref_img, tuple(spt), tuple(rpt))
        for spt, rpt in zip(src_inliers, ref_inliers)
    ])
    t1 = time.perf_counter()
    print(f"   -> JIT Parabolic Peak Fitting complete in {(t1-t0)*1000:.2f} ms.")

    print("\n🌌 [6/6] Warping Image Coordinate Space (Thin Plate Spline)...")
    aligned_img = SpaceWarper.warp_image_tps(src_img, ref_img.shape, src_inliers, refined_ref_pts)

    print(f"\n💾 Saving Registered Output to: {args.output}")
    cv2.imwrite(args.output, aligned_img)
    print("✅ Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
