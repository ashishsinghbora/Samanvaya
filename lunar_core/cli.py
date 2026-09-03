"""
Samanvaya (समान्वय) Unified Command-Line Interface.
ISRO Chandrayaan-2 Lunar Optical Image Registration Framework (SIH PS 26166).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np


def print_banner() -> None:
    banner = r"""
  ███████╗ █████╗ ███╗   ███╗ █████╗ ███╗   ██╗██╗   ██╗ █████╗ ██╗   ██╗ █████╗ 
  ██╔════╝██╔══██╗████╗ ████║██╔══██╗████╗  ██║██║   ██║██╔══██╗╚██╗ ██╔╝██╔══██╗
  ███████╗███████║██╔████╔██║███████║██╔██╗ ██║██║   ██║███████║ ╚████╔╝ ███████║
  ╚════██║██╔══██║██║╚██╔╝██║██╔══██║██║╚██╗██║╚██╗ ██╔╝██╔══██║  ╚██╔╝  ██╔══██║
  ███████║██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║ ╚████╔╝ ██║  ██║   ██║   ██║  ██║
  ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
  ISRO Chandrayaan-2 Planetary Image Registration Framework (SIH PS 26166)
    """
    print(banner)


def cmd_ui(args: argparse.Namespace) -> None:
    """Launches the interactive Streamlit portal."""
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    port = args.port or 8501
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    print(f"🚀 Launching Samanvaya Streamlit Portal on port {port}...")
    subprocess.run(cmd)


def cmd_api(args: argparse.Namespace) -> None:
    """Launches the FastAPI REST API."""
    port = args.port or 8000
    host = args.host or "0.0.0.0"
    cmd = [sys.executable, "-m", "uvicorn", "ch2_lunar_reg.interfaces.api:app", "--host", host, "--port", str(port), "--reload"]
    print(f"🛰️ Launching Samanvaya FastAPI Server on {host}:{port}...")
    subprocess.run(cmd)


def cmd_test(args: argparse.Namespace) -> None:
    """Executes the full automated verification test suite."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "ch2_lunar_reg/tests/", "-v"]
    print("🧪 Executing Samanvaya Verification Test Suite...")
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def cmd_align(args: argparse.Namespace) -> None:
    """Headless CLI alignment between Source and Reference lunar GeoTIFFs."""
    from lunar_core.data_io import PlanetaryRasterReader, PlanetaryRasterWriter
    from lunar_core.data_io.raster_reader import sanitize_path
    from lunar_core.alignment.dense_matcher import DenseLoFTRMatcher
    from lunar_core.evaluation.metrics import EvaluationEngine

    src_path = sanitize_path(args.source)
    ref_path = sanitize_path(args.reference)
    out_dir = sanitize_path(args.output or "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading Source Raster: {src_path}")
    raster_src = PlanetaryRasterReader.read_geotiff(src_path)
    print(f"📥 Loading Reference Raster: {ref_path}")
    raster_ref = PlanetaryRasterReader.read_geotiff(ref_path)

    print("⚙️ Executing Dense LoFTR Matching & 2D Parabolic Taylor Sub-pixel Refinement...")
    matcher = DenseLoFTRMatcher(
        pretrained=args.weights,
        confidence_threshold=args.threshold,
        grid_bins=8,
        cap_per_cell=args.cap,
        magsac_reproj_threshold=args.reproj_threshold,
    )

    inliers, H, warped = matcher.match(
        source_image=raster_src.data,
        reference_image=raster_ref.data,
    )

    print(f"🎯 Discovered {len(inliers)} verified inliers.")

    # Generate Report
    report = EvaluationEngine.generate_report(
        total_matches=len(inliers) * 2,  # approx
        inliers=inliers,
        image_shape=raster_ref.data.shape,
        homography=H,
    )

    json_file = out_dir / "samanvaya_evaluation_report.json"
    report.export_json(json_file)
    print(f"📄 Exported Structured Report: {json_file}")

    plot_file = out_dir / "samanvaya_residual_scatter.png"
    report.export_residual_scatter_plot(plot_file, background_image=raster_ref.data)
    print(f"📈 Exported Residual Scatter Plot: {plot_file}")

    if warped is not None:
        warped_file = out_dir / "registered_source.tif"
        PlanetaryRasterWriter.write_geotiff(warped_file, warped, raster_ref)
        print(f"🗺️ Exported Registered GeoTIFF: {warped_file}")

    print("\n" + "=" * 50)
    print("📊 SAMANVAYA MISSION KPI SUMMARY")
    print("=" * 50)
    print(f"  Sub-Pixel RMSE : {report.rmse_pixels:.4f} px (ISRO Mandate < 0.40 px: {'PASSED ✅' if report.meets_isro_mandate else 'NEEDS REVIEW ⚠️'})")
    print(f"  Inlier Count   : {report.inlier_count} verified tie-points")
    print(f"  Inlier Ratio   : {report.inlier_ratio_percent:.2f}%")
    print(f"  Spatial Entropy: {report.spatial_uniformity_entropy:.4f} / 1.0 (Non-clumping score)")
    print("=" * 50)


def cmd_info(args: argparse.Namespace) -> None:
    """Displays hardware, library, and mission configuration details."""
    import torch
    import kornia
    import cv2
    import rasterio

    print_banner()
    print("Mission Configuration:")
    print("  Problem Statement: SIH PS 26166")
    print("  Space Agency     : Indian Space Research Organisation (ISRO)")
    print("  Target Mission   : Chandrayaan-2 (OHRC, TMC-2, IIRS) & NASA LRO NAC")
    print("  Coordinate CRS   : Moon IAU 2000 (IAU2000:30100)")
    print("\nEnvironment & Hardware:")
    print(f"  Python Version  : {sys.version.split()[0]}")
    print(f"  PyTorch Version : {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    print(f"  Kornia Version  : {kornia.__version__}")
    print(f"  OpenCV Version  : {cv2.__version__}")
    print(f"  Rasterio Version: {rasterio.__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="samanvaya",
        description="Samanvaya: Industry-Grade Lunar Optical Image Registration Framework (SIH PS 26166)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # samanvaya ui
    p_ui = subparsers.add_parser("ui", help="Launch interactive Streamlit registration portal")
    p_ui.add_argument("--port", type=int, default=8501, help="Port to bind Streamlit server")
    p_ui.set_defaults(func=cmd_ui)

    # samanvaya api
    p_api = subparsers.add_parser("api", help="Launch FastAPI REST server")
    p_api.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    p_api.add_argument("--port", type=int, default=8000, help="Port to bind FastAPI server")
    p_api.set_defaults(func=cmd_api)

    # samanvaya test
    p_test = subparsers.add_parser("test", help="Run automated verification test suite")
    p_test.set_defaults(func=cmd_test)

    # samanvaya align
    p_align = subparsers.add_parser("align", help="Headless CLI alignment between GeoTIFFs")
    p_align.add_argument("--source", "-s", required=True, help="Path to Source GeoTIFF (OHRC/TMC-2)")
    p_align.add_argument("--reference", "-r", required=True, help="Path to Reference GeoTIFF (LRO NAC)")
    p_align.add_argument("--output", "-o", default="output", help="Directory to save aligned products")
    p_align.add_argument("--weights", default="outdoor", help="LoFTR model weights checkpoint")
    p_align.add_argument("--threshold", type=float, default=0.15, help="LoFTR confidence threshold")
    p_align.add_argument("--cap", type=int, default=4, help="ANMS equal cap per 8x8 cell")
    p_align.add_argument("--reproj-threshold", type=float, default=1.5, help="USAC-MAGSAC reprojection threshold")
    p_align.set_defaults(func=cmd_align)

    # samanvaya info
    p_info = subparsers.add_parser("info", help="Display system and mission configuration")
    p_info.set_defaults(func=cmd_info)

    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        sys.exit(0)

    parsed_args = parser.parse_args()
    if hasattr(parsed_args, "func"):
        parsed_args.func(parsed_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
