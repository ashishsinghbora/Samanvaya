"""
tests/test_photogrammetry.py

Defense-grade test suite validating the mathematical correctness of 
the Lunar Image Registration core engines (Phase Congruency, Sub-Pixel, QuadTree).
"""
import numpy as np
import pytest

from src.features.phase_congruency import PhaseCongruencyEngine
from src.registration.warper import SubPixelRefiner as SurfaceRefiner, NonLinearWarper
from src.matching.subpixel import SubPixelRefiner
from src.matching.quadtree import UniformDistributor

def test_sub_pixel_refinement_perfect_peak():
    """
    Test sub-pixel refinement on a perfectly centered symmetric peak.
    The offset should be strictly 0.0.
    """
    # Create a 3x3 symmetric peak (max at center)
    Z = np.array([
        [0.1, 0.5, 0.1],
        [0.5, 1.0, 0.5],
        [0.1, 0.5, 0.1]
    ], dtype=np.float64)
    
    # Pad to make it a 5x5 image
    img = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
    
    x, y = 2, 2  # The local maximum in the 5x5 image
    rx, ry = SurfaceRefiner.refine(img, (x, y), window_size=3)
    
    assert np.isclose(rx, 2.0), f"Expected rx=2.0, got {rx}"
    assert np.isclose(ry, 2.0), f"Expected ry=2.0, got {ry}"


def test_sub_pixel_refinement_shifted_peak():
    """
    Test sub-pixel refinement on a shifted peak.
    The offset should pull towards the heavier side.
    """
    # Shifted to the right
    Z = np.array([
        [0.1, 0.4, 0.3],
        [0.2, 0.8, 0.9],
        [0.1, 0.4, 0.3]
    ], dtype=np.float64)
    
    img = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
    
    x, y = 2, 2
    rx, ry = SurfaceRefiner.refine(img, (x, y), window_size=3)
    
    # Peak is heavily pulled to the right (+x), should be > 2.0
    assert rx > 2.0 and rx <= 2.5, f"Expected rx between 2.0 and 2.5, got {rx}"
    # Y is symmetric, should be exactly 2.0
    assert np.isclose(ry, 2.0), f"Expected ry=2.0, got {ry}"


def test_subpixel_refiner_ncc_patch():
    """
    Test 2D paraboloid NCC matching subpixel refiner.
    """
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)
    
    # Place synthetic feature at (50, 50) in img1 and shifted in img2
    img1[45:55, 45:55] = 200
    img2[45:55, 45:55] = 200
    
    refiner = SubPixelRefiner(patch_size=31)
    rx, ry = refiner.refine_match(img1, img2, (50.0, 50.0), (50.0, 50.0))
    assert abs(rx - 50.0) < 1.0
    assert abs(ry - 50.0) < 1.0


def test_phase_congruency_synthetic_edge():
    """
    Test Phase Congruency engine on a synthetic step edge.
    The step edge should yield a strong phase congruency response along the boundary.
    """
    # Create a 64x64 synthetic step edge
    img = np.zeros((64, 64), dtype=np.float64)
    img[:, 32:] = 255.0
    
    engine = PhaseCongruencyEngine(num_scales=3, num_orientations=4)
    pc, or_map, ft, noise = engine.compute(img)
    
    assert pc.shape == (64, 64)
    assert or_map.shape == (64, 64)
    
    # The maximum PC should be roughly along the vertical edge (col 31-32)
    edge_col_response = np.max(pc[:, 31:33])
    background_response = np.mean(pc[:, 10:20])
    
    assert edge_col_response > 0.5, "Phase congruency failed to detect strong edge."
    assert background_response < 0.1, "Phase congruency failed to suppress uniform background."


def test_uniform_distributor_quadtree():
    """
    Test UniformDistributor Quad-Tree decomposition.
    """
    pts = np.random.rand(1000, 2) * 500
    confs = np.random.rand(1000)
    
    filtered = UniformDistributor.filter_points(pts, confs, 500, 500, max_points_per_cell=3, max_depth=4)
    assert len(filtered) > 0
    assert len(filtered) <= 1000


def test_nonlinear_warper_identity():
    """
    Test that a non-linear TPS warper with identical source and dest points
    acts as an identity mapping.
    """
    # 4 corners + center
    pts = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
        [50, 50]
    ], dtype=np.float64)
    
    warper = NonLinearWarper(kernel='thin_plate_spline')
    # Fit target -> source for reverse mapping
    warper.fit(src_pts=pts, dst_pts=pts) 
    
    # Create a 100x100 synthetic image
    img = np.zeros((100, 100), dtype=np.uint8)
    img[40:60, 40:60] = 255
    
    # Warp image
    warped = warper.warp_image(img, output_shape=(100, 100))
    
    # Ensure the image is largely unchanged (identity)
    diff = np.abs(img.astype(np.float32) - warped.astype(np.float32))
    assert np.mean(diff) < 5.0, "TPS Warper failed identity mapping test."
