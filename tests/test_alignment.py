# tests/test_alignment.py
import numpy as np
from depth_alignment.alignment import align_depths

def test_align_depths():
    # Create synthetic data for testing
    true_scale, true_bias = 2.0, 5.0
    depth_estimates = np.array([10, 20, 30, 40])
    colmap_depths = depth_estimates * true_scale + true_bias

    # Run alignment
    scale, bias = align_depths(colmap_depths, depth_estimates)
    assert np.isclose(scale, true_scale), "Scale mismatch"
    assert np.isclose(bias, true_bias), "Bias mismatch"
