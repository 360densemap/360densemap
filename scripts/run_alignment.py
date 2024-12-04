# scripts/run_alignment.py
from src.alignment import align_depths, apply_scale_bias
from src import utils
from src.utils import (load_depth_map, generate_points3D_file, world2cam, cam2world)
from src.utils import generate_new_points_file_single_line
from src.visualize import (visualize_features_on_image, visualize_depth_alignment, visualize_3d_points, visualize_3d_depth_map_comparison)
import os
import numpy as np
import cv2
import time

# Dynamically calculate the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Paths to the input files
CAMERAS_FILE = os.path.join(PROJECT_ROOT, '/path/to/cameras.txt')
IMAGES_FILE = os.path.join(PROJECT_ROOT, '/path/to/Input/images.txt')
POINTS3D_FILE = os.path.join(PROJECT_ROOT, '/path/to/Input/points3D.txt')

def main():
    # Start the timer
    start_time = time.time()
    # Load COLMAP data
    cameras = utils.read_cameras(CAMERAS_FILE)
    images = utils.read_images(IMAGES_FILE)
    points3D = utils.read_points3D(POINTS3D_FILE)

    # Specify the image to process
    image_id = 59  # Replace with the actual ID of your chosen image
    chosen_image = images[image_id]

    # Extract 2D features and their corresponding 3D points
    features = utils.get_features_for_camera(images, points3D, image_id)
    features_3D = [point3D for _, _, point3D in features]  # Extract 3D points only

    # Get the camera extrinsic parameters
    chosen_image = images[image_id]
    quaternion = chosen_image["quaternion"]
    translation = chosen_image["translation"]

    # Transform the 3D points to the camera frame
    transformed_points  = utils.world2cam(features_3D, quaternion, translation)

    # Visualize 3D points seen by the camera
    #visualize_3d_points(transformed_points)

    # Associate 2D feature coordinates with their corresponding 3D points in the camera frame
    camera_features = [(x, y, point_cam) for (x, y, _), point_cam in zip(features, transformed_points)]

    # Compute spherical coordinates and metric depth
    spherical_with_depth = utils.spherical_coordinates_and_depth(camera_features)

    # Extract metric depths for alignment
    metric_depths = [(feature["x"], feature["y"], feature["depth"]) for feature in spherical_with_depth]

    print("Metric depths extracted for alignment:")
    for metric_depth in metric_depths:
        print(metric_depth)


    # Visualize 2D features
    features_2d = [(x, y) for x, y, _ in features]
    depth_map_path = '/path/to/depth_images/59.png'  
    #visualize_features_on_image(depth_map_path,features_2d)

    depth_map = load_depth_map(depth_map_path)

    # Match feature metric depths with estimated depths in the depth map
    colmap_depths, depth_estimates = utils.match_features_with_depth_map(metric_depths, depth_map)

    # Align depths using least squares to find optimal scale and bias
    scale, bias = align_depths(np.array(colmap_depths), np.array(depth_estimates))
    print(f"Scale: {scale}, Bias: {bias}")

    # Apply scale and bias to the depth map for final alignment
    aligned_depth = apply_scale_bias(depth_map, scale, bias)
    end_time = time.time()
    # Visualize aligned depth map and differences
    #visualize_depth_alignment(depth_map, aligned_depth)
     # Load the depth map and RGB image

    rgb_image_path = '/path/to/depth_images/059.jpg'

    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    #visualize_3d_depth_map_comparison( depth_map, aligned_depth, rgb_image, width=rgb_image.shape[1], height=rgb_image.shape[0], equ_cx=rgb_image.shape[1] / 2, equ_cy=rgb_image.shape[0] / 2)

    new_points2d = generate_points3D_file(
        aligned_depth_map=aligned_depth,
        original_features=features_2d,
        rgb_image=rgb_image,
        output_file="/path/to/newpoint3d.txt",
        start_id=5700,
        error_value=1.0,
        image_id=59,
        quaternion=quaternion,
        translation=translation,
        start_point2d_idx=704
    )
    # Generate the new points file
    generate_new_points_file_single_line(new_points2d, "/path/to/newimage.txt")

    elapsed_time = end_time - start_time
    print(f"Process completed in {elapsed_time:.2f} seconds.")
if __name__ == "__main__":
    main()
