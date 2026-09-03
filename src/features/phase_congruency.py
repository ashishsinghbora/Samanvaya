"""
Sun-Angle Invariant Feature Extractor using Phase Congruency.
Computes structural crater rims and geological features independent of 
solar incidence angle and shadow depth by analyzing phase alignment in the frequency domain.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional

class PhaseCongruencyEngine:
    """
    Computes Phase Congruency of lunar orbital imagery using 2D Log-Gabor filters.
    """

    def __init__(self, scales: int = 4, orientations: int = 6, 
                 min_wave_length: int = 3, mult: float = 2.0, 
                 sigma_on_f: float = 0.55):
        """
        Initializes the Log-Gabor filter bank parameters.
        
        Args:
            scales: Number of wavelet scales (frequencies).
            orientations: Number of filter orientations.
            min_wave_length: Wavelength of the smallest scale filter.
            mult: Scaling factor between successive filters.
            sigma_on_f: Ratio of the standard deviation of the Gaussian 
                        describing the log Gabor filter's transfer function 
                        in the frequency domain to the filter center frequency.
        """
        self.scales = scales
        self.orientations = orientations
        self.min_wave_length = min_wave_length
        self.mult = mult
        self.sigma_on_f = sigma_on_f

    def _create_log_gabor_filter(self, rows: int, cols: int) -> list:
        """
        Constructs the frequency domain Log-Gabor filter bank.
        """
        # Create normalized frequency grid [-0.5, 0.5]
        u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols), 
                             np.linspace(-0.5, 0.5, rows))
        
        # Center origin at (0,0) for FFT alignment
        u1 = ifftshift(u1)
        u2 = ifftshift(u2)
        radius = np.sqrt(u1**2 + u2**2)
        radius[0, 0] = 1.0  # Prevent divide by zero at DC component
        
        theta = np.arctan2(-u2, u1)
        
        # Pre-compute angular filter components
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        
        filters = []
        for o in range(self.orientations):
            angl = o * np.pi / self.orientations
            # Wavelength of angular distance
            ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
            dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
            dtheta = np.abs(np.arctan2(ds, dc))
            
            # Angular Gaussian mask
            spread = np.exp((-dtheta**2) / (2 * (np.pi / self.orientations / 1.5)**2))
            
            for s in range(self.scales):
                wavelength = self.min_wave_length * (self.mult ** s)
                center_freq = 1.0 / wavelength
                
                # Log-Gabor radial component
                log_gabor = np.exp((-(np.log(radius / center_freq))**2) / (2 * np.log(self.sigma_on_f)**2))
                log_gabor[0, 0] = 0.0 # DC component must be zero
                
                filters.append(log_gabor * spread)
                
        return filters

    def compute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the Phase Congruency transform on a lunar image tile.
        
        Args:
            image: 2D NumPy array representing the grayscale lunar raster tile.
            
        Returns:
            Tuple containing:
                - pc_image: 2D Phase Congruency map normalized [0, 1].
                - orientation_map: Dominant structural orientation at each pixel.
        """
        if image.ndim != 2:
            raise ValueError("Input image must be a 2D grayscale array.")
            
        rows, cols = image.shape
        image_fft = fft2(image)
        
        filters = self._create_log_gabor_filter(rows, cols)
        
        sum_E = np.zeros((rows, cols))
        sum_O = np.zeros((rows, cols))
        sum_An = np.zeros((rows, cols))
        
        idx = 0
        for o in range(self.orientations):
            sum_E_orient = np.zeros((rows, cols))
            sum_O_orient = np.zeros((rows, cols))
            sum_An_orient = np.zeros((rows, cols))
            
            for s in range(self.scales):
                filter_2d = filters[idx]
                idx += 1
                
                # Apply filter in frequency domain and return to spatial
                EO_fft = image_fft * filter_2d
                EO = ifft2(EO_fft)
                
                E = np.real(EO)
                O = np.imag(EO)
                An = np.sqrt(E**2 + O**2)
                
                sum_E_orient += E
                sum_O_orient += O
                sum_An_orient += An
            
            sum_E += sum_E_orient
            sum_O += sum_O_orient
            sum_An += sum_An_orient

        # Calculate localized Phase Congruency (Energy / Amplitude)
        energy = np.sqrt(sum_E**2 + sum_O**2)
        epsilon = 1e-4 # Prevent division by zero
        pc_image = np.maximum(energy - epsilon, 0) / (sum_An + epsilon)
        
        # Calculate dominant orientation map
        orientation_map = np.arctan2(sum_O, sum_E)
        
        return pc_image, orientation_map
