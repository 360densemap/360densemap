import cv2
import numpy as np
import torch

def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

def remap_to_spherical(keypoints, width, height):
    spherical_keypoints = []
    for x, y in keypoints:
        theta = (x / width) * 2 * np.pi  # Map x to longitude
        phi = (y / height) * np.pi       # Map y to latitude
        spherical_keypoints.append([theta, phi, 1.0, 0.0])  # Format compatible with COLMAP database
    return np.array(spherical_keypoints, dtype=np.float32)