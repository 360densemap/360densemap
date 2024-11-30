# scripts/run_alignment.py
from src.alignment import align_depths, apply_scale_bias
from src import utils
from src.visualize import (visualize_features_on_image, visualize_depth_alignment, visualize_spherical_with_depth)

import os
import numpy as np


# Dynamically calculate the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Paths to the input files
CAMERAS_FILE = os.path.join(PROJECT_ROOT, 'depth_alignment/Input/cameras.txt')
IMAGES_FILE = os.path.join(PROJECT_ROOT, 'depth_alignment/Input/images.txt')
POINTS3D_FILE = os.path.join(PROJECT_ROOT, 'depth_alignment/Input/points3D.txt')


def main():
    # Load COLMAP data
    cameras = utils.read_cameras(CAMERAS_FILE)
    images = utils.read_images(IMAGES_FILE)
    points3D = utils.read_points3D(POINTS3D_FILE)

    # Group images by pose (assuming filename prefix method)
    grouped_images = utils.group_images_by_pose(images, grouping_criterion="name_prefix")

    # Debug: Print all generated pose IDs
    #print("Generated pose IDs:", list(grouped_images.keys()))

    # Choose a pose ID to aggregate (e.g., 'P1180141')
    pose_id = 'image_0062'

    # Aggregate features for the chosen pose
    aggregated_features = utils.aggregate_features_for_pose(images, points3D, grouped_images, pose_id)

    # Visualize 2D features
    features_2d = [(x, y) for x, y, _ in aggregated_features]
    visualize_features_on_image('data/depth_map.png', features_2d)
    
    
    # Convert aggregated features to spherical coordinates and compute depth simultaneously
    spherical_with_depth = []
    for x, y, (X, Y, Z) in aggregated_features:
        # Compute depth (Euclidean norm) directly
        depth = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Convert to spherical coordinates (phi, theta)
        theta = np.arccos(Z / depth)  # Inclination angle
        phi = np.arctan2(Y, X)        # Azimuth angle

        # Store spherical coordinates with depth
        spherical_with_depth.append({"phi": phi, "theta": theta, "depth": depth})

    # Visualize spherical coordinates with depths
    visualize_spherical_with_depth(spherical_with_depth)

    # Print the spherical coordinates and depths
    print("Spherical coordinates with depths for each feature:")
    for feature in spherical_with_depth:
        print(feature)

    # Initialize equirectangular converter
    equirectangular_converter = utils.EquirectangularToSpherical('data/depth_map.png')

    # Load depth estimates from .npy file
    depth_map = utils.load_depth_map('data/depth_estimate.npy')

    # Match feature metric depths with estimated depths in the depth map
    matched_depths = utils.match_features_with_depth_map(spherical_with_depth, depth_map, equirectangular_converter)

    # Separate metric depths and depth estimates for alignment
    colmap_depths, depth_estimates = zip(*matched_depths)
    colmap_depths = np.array(colmap_depths)
    depth_estimates = np.array(depth_estimates)

    # Align depths using least squares to find optimal scale and bias
    scale, bias = align_depths(colmap_depths, depth_estimates)
    print(f"Scale: {scale}, Bias: {bias}")

    # Apply scale and bias to the depth map for final alignment
    aligned_depth = apply_scale_bias(depth_map, scale, bias)
    print("Aligned depth map:", aligned_depth)

    # Visualize aligned depth map and differences
    visualize_depth_alignment(depth_map, aligned_depth)

if __name__ == "__main__":
    main()
