# depth_alignment/utils.py
import numpy as np
import cv2

def load_depth_map(filepath):
    # Example of loading depth map (assumed to be a .npy file)
    return np.load(filepath)

def load_colmap_depths(filepath):
    # Load COLMAP feature depths, assume simple text or numpy format
    return np.loadtxt(filepath)

class EquirectangularToSpherical:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        self._equ_cx = (self._width - 1) / 2.0
        self._equ_cy = (self._height - 1) / 2.0

    def pixel_to_spherical(self, x, y):
        """
        Convert a pixel (x, y) in an equirectangular image to spherical coordinates (phi, theta).

        Parameters:
        - x (float): x-coordinate of the pixel.
        - y (float): y-coordinate of the pixel.

        Returns:
        - phi (float): Longitude in degrees, ranging from -180 to 180.
        - theta (float): Latitude in degrees, ranging from -90 to 90.
        """
        phi = (x - self._equ_cx) / self._equ_cx * 180
        theta = -(y - self._equ_cy) / self._equ_cy * 90
        return phi, theta
