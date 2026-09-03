"""
src/matching/asift.py

Affine-SIFT (ASIFT) multi-scale pyramidal matcher simulation.
Simulates viewpoint changes (tilt and rotation) to establish 
correspondences across extreme scale variations (e.g., OHRC to IIRS).
"""
import cv2
import numpy as np

class ASIFTMatcher:
    def __init__(self, max_tilt: int = 4, tilt_step: float = 1.414):
        self.max_tilt = max_tilt
        self.tilt_step = tilt_step
        self.detector = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
        
    def _generate_affine_simulations(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generates simulated affine distortions of the image by tilting and rotating.
        Returns a list of (warped_image, affine_transform_matrix).
        """
        simulations = []
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        for tilt in np.arange(1.0, self.max_tilt + 0.1, self.tilt_step):
            # Calculate rotation step based on tilt (72 degrees / tilt)
            if tilt == 1.0:
                rotations = [0]
            else:
                rot_step = 72.0 / tilt
                rotations = np.arange(0, 180, rot_step)

            for phi in rotations:
                # Rotation matrix
                M_rot = cv2.getRotationMatrix2D(center, phi, 1.0)
                
                # Tilt matrix (scale along x)
                t = 1.0 / tilt
                M_tilt = np.array([[t, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

                # Combine transforms
                M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
                M_affine = M_tilt @ M_rot_3x3
                
                warped = cv2.warpPerspective(
                    image, 
                    M_affine, 
                    (w, h), 
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                simulations.append((warped, M_affine))
                
        return simulations

    def detect_and_compute(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregates features across all affine simulations.
        """
        all_keypoints = []
        all_descriptors = []
        
        simulations = self._generate_affine_simulations(image)
        for warped, M_affine in simulations:
            kps, des = self.detector.detectAndCompute(warped, None)
            if kps and des is not None:
                # Invert transform to map keypoints back to original image space
                M_inv = np.linalg.inv(M_affine)
                
                for i, kp in enumerate(kps):
                    # Transform (x, y) back
                    pt = np.array([kp.pt[0], kp.pt[1], 1.0])
                    pt_orig = M_inv @ pt
                    
                    # Create new keypoint in original space
                    kp_orig = cv2.KeyPoint(
                        x=pt_orig[0] / pt_orig[2],
                        y=pt_orig[1] / pt_orig[2],
                        size=kp.size,
                        angle=kp.angle,
                        response=kp.response,
                        octave=kp.octave,
                        class_id=kp.class_id
                    )
                    all_keypoints.append(kp_orig)
                    all_descriptors.append(des[i])
                    
        return all_keypoints, np.array(all_descriptors)
