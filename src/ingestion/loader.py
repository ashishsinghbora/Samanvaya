"""
src/ingestion/loader.py

Geospatial Image Loader.
Handles reading large raster files securely using GDAL/Rasterio, 
preventing Out-of-Memory (OOM) errors via dynamic downsampling if needed.
"""
import rasterio
import numpy as np
from pathlib import Path
from src.core.exceptions import InvalidFileFormatError

class RasterLoader:
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Raster file not found: {self.filepath}")

    def load_raster(self, max_dim: int = 4096) -> tuple[np.ndarray, rasterio.profiles.Profile]:
        """
        Loads raster data. If dimensions exceed max_dim, it downsamples
        to prevent memory exhaustion.
        """
        try:
            with rasterio.open(self.filepath) as src:
                profile = src.profile
                width = src.width
                height = src.height

                # Calculate downsample factor if too large
                scale = 1.0
                if width > max_dim or height > max_dim:
                    scale = min(max_dim / width, max_dim / height)
                    
                new_width = int(width * scale)
                new_height = int(height * scale)

                data = src.read(
                    1, # Read first band
                    out_shape=(1, new_height, new_width),
                    resampling=rasterio.enums.Resampling.bilinear
                )
                
                # Update profile to reflect new shape/transform
                transform = src.transform * src.transform.scale(
                    (src.width / data.shape[-1]),
                    (src.height / data.shape[-2])
                )
                
                profile.update({
                    'height': new_height,
                    'width': new_width,
                    'transform': transform,
                    'dtype': data.dtype
                })
                
                return data, profile
        except rasterio.errors.RasterioIOError as e:
            raise InvalidFileFormatError(f"Failed to load raster {self.filepath}: {str(e)}")
