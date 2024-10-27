# scripts/run_alignment.py
from depth_alignment.alignment import align_depths, apply_scale_bias
from depth_alignment import utils
import numpy as np

def main():
    # Load COLMAP data
    cameras = utils.read_cameras('data/cameras.txt')
    images = utils.read_images('data/images.txt')
    points3D = utils.read_points3D('data/points3D.txt')

    # Group images by pose (assuming filename prefix method)
    grouped_images = utils.group_images_by_pose(images, grouping_criterion="name_prefix")

    # Choose a pose ID to aggregate (e.g., 'P1180141')
    pose_id = 'P1180141'

    # Aggregate features for the chosen pose
    aggregated_features = utils.aggregate_features_for_pose(images, points3D, grouped_images, pose_id)

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

if __name__ == "__main__":
    main()
