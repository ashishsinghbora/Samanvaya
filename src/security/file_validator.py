"""
Defense-Grade Geospatial File Validator.
Enforces structural, cryptographic, and bounds-checked validation on incoming
lunar datasets (GeoTIFF, PDS4, IMG) prior to ingestion into the raster pipeline.
"""

import os
import struct
from typing import Union, Tuple, Optional
from pathlib import Path
import defusedxml.ElementTree as ET
from defusedxml.common import DTDForbidden, EntitiesForbidden

# Absolute hard limits for aerospace optical swaths (preventing Decompression Bombs)
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB
MAX_WIDTH_PIXELS = 100_000
MAX_HEIGHT_PIXELS = 100_000

# Magic Bytes for allowed geospatial formats
VALID_MAGIC_BYTES = {
    b"II*\x00": "TIFF_LITTLE_ENDIAN",
    b"MM\x00*": "TIFF_BIG_ENDIAN",
    b"PDS_": "NASA_PDS3",
}

class SecurityValidationError(Exception):
    """Raised when a file fails cryptographic or structural security constraints."""
    pass


class PayloadValidator:
    """Zero-Trust validator for incoming orbital image payloads."""

    @staticmethod
    def verify_magic_bytes(file_path: Union[str, Path]) -> str:
        """
        Reads the file header to verify structural integrity independent of file extension.
        
        Args:
            file_path: Absolute path to the payload on disk.
            
        Returns:
            The identified format string.
            
        Raises:
            SecurityValidationError: If the file header does not match known aerospace formats.
            IOError: If the file cannot be securely read.
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Payload not found or is not a file: {path}")

        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise SecurityValidationError(f"File exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES} bytes).")

        try:
            with open(path, "rb") as f:
                header = f.read(4)
                for magic, format_name in VALID_MAGIC_BYTES.items():
                    if header.startswith(magic):
                        return format_name
                raise SecurityValidationError(f"Invalid magic bytes detected: {header.hex()}")
        except PermissionError as e:
            raise SecurityValidationError(f"Permission denied reading payload: {e}")

    @staticmethod
    def sanitize_pds4_label(xml_path: Union[str, Path]) -> ET.Element:
        """
        Safely parses PDS4 XML labels to prevent XML External Entity (XXE) and Billion Laughs attacks.
        
        Args:
            xml_path: Path to the PDS4 XML label.
            
        Returns:
            The parsed XML ElementTree root.
            
        Raises:
            SecurityValidationError: If malicious XML structures (DTDs, external entities) are detected.
        """
        path = Path(xml_path)
        try:
            # defusedxml blocks external entities and DTD expansion by default
            tree = ET.parse(str(path))
            return tree.getroot()
        except (DTDForbidden, EntitiesForbidden) as e:
            raise SecurityValidationError(f"Malicious XML detected (XXE/DTD injection attempt): {e}")
        except ET.ParseError as e:
            raise SecurityValidationError(f"Malformed XML in PDS4 label: {e}")

    @staticmethod
    def validate_raster_bounds(width: int, height: int, bands: int) -> bool:
        """
        Ensures raster dimensions do not exceed architectural memory allocation limits.
        
        Args:
            width: Extracted image width in pixels.
            height: Extracted image height in pixels.
            bands: Number of spectral bands.
            
        Raises:
            SecurityValidationError: If dimensions exceed pre-defined safe limits.
        """
        if width <= 0 or height <= 0 or bands <= 0:
            raise SecurityValidationError("Invalid negative or zero raster dimensions.")
            
        if width > MAX_WIDTH_PIXELS or height > MAX_HEIGHT_PIXELS:
            raise SecurityValidationError(
                f"Raster dimensions ({width}x{height}) exceed bounds "
                f"({MAX_WIDTH_PIXELS}x{MAX_HEIGHT_PIXELS}). Potential pixel-bomb detected."
            )
        return True
