# depth_alignment/alignment.py
import numpy as np
from scipy.optimize import least_squares

def align_depths(colmap_depths, depth_estimates):
    # Construct the design matrix `A` and target vector `b`
    A = np.vstack([depth_estimates, np.ones(len(depth_estimates))]).T
    b = colmap_depths

    # Solve using SciPy's least squares
    result = least_squares(lambda x: A @ x - b, x0=[1.0, 0.0], loss='huber', f_scale=1.0)
    scale, bias = result.x
    return scale, bias

def apply_scale_bias(depth_map, scale, bias):
    return depth_map * scale + bias
