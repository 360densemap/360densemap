# depth_alignment/utils.py
import numpy as np
import cv2
from src.alignment import align_depths, apply_scale_bias

def read_cameras(file_path):
    cameras = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split()
            camera_id = int(data[0])
            model = data[1]
            width = int(data[2])
            height = int(data[3])
            params = list(map(float, data[4:]))
            cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    return cameras

def read_images(file_path):
    images = {}
    with open(file_path, 'r') as file:
        line = file.readline()
        while line:
            if line.startswith("#"):
                line = file.readline()
                continue

            # Parse the first line (metadata)
            data = line.strip().split()
            image_id = int(data[0])
            quaternion = list(map(float, data[1:5]))
            translation = list(map(float, data[5:8]))
            camera_id = int(data[8])
            name = data[9]

            # Parse the second line (POINTS2D[])
            line = file.readline().strip()
            keypoints = []
            if line:  # Ensure the line is not empty
                keypoints_raw = line.split()
                for i in range(0, len(keypoints_raw), 3):
                    if i + 2 < len(keypoints_raw):  # Ensure there are enough values to unpack
                        x, y, point3D_id = float(keypoints_raw[i]), float(keypoints_raw[i + 1]), int(keypoints_raw[i + 2])
                        keypoints.append((x, y, point3D_id))

            # Store the parsed data
            images[image_id] = {
                "quaternion": quaternion,
                "translation": translation,
                "camera_id": camera_id,
                "name": name,
                "keypoints": keypoints,
            }

            # Read the next line
            line = file.readline()
    return images


def read_points3D(file_path):
    points3D = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            data = line.strip().split()
            point3D_id = int(data[0])
            xyz = list(map(float, data[1:4]))
            rgb = list(map(int, data[4:7]))
            error = float(data[7])
            track = [tuple(map(int, data[i:i+2])) for i in range(8, len(data), 2)]
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points3D

def get_features_for_camera(image_data, points3D, chosen_image_id):
    chosen_image = image_data[chosen_image_id]
    camera_features = []

    # Extract features observed by the chosen image
    for keypoint in chosen_image['keypoints']:
        x, y, point3D_id = keypoint
        if point3D_id == -1:
            continue  # Ignore keypoints without a corresponding 3D point
        point3D = points3D[point3D_id]["xyz"]
        camera_features.append((x, y, point3D))

    return camera_features


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])

def spherical_coordinates_and_depth(features):
    """
    Compute spherical coordinates (phi, theta) and metric depth for 3D points in the camera frame.

    Parameters:
    - features: List of (x, y, (X, Y, Z)) tuples, where (X, Y, Z) are 3D points in the camera frame.

    Returns:
    - List of dictionaries containing phi, theta, depth, x, and y for each feature.
    """
    spherical_with_depth = []
    for x, y, (X, Y, Z) in features:
        # Calculate metric depth (distance from the origin in the camera frame)
        depth = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Calculate spherical coordinates (phi, theta)
        theta = np.arccos(Z / depth) if depth != 0 else 0  # Angle from Z-axis
        phi = np.arctan2(Y, X)  # Angle in the XY-plane
        
        # Append the data, including 2D feature coordinates
        spherical_with_depth.append({
            "phi": phi, 
            "theta": theta, 
            "depth": depth, 
            "x": x, 
            "y": y
        })
    return spherical_with_depth


def world2cam(features_3D, quaternion, translation):
    """
    Transform 3D points from world coordinates to the camera coordinates of a specific image.

    Parameters:
    - features_3D: List of 3D points in world coordinates (filtered from `get_features_for_camera`).
    - quaternion: Quaternion defining the camera's orientation.
    - translation: Translation vector defining the camera's position.

    Returns:
    - points_camera: List of 3D points in the chosen image's camera coordinates.
    """
    # Convert the quaternion to a rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    translation_vector = np.array(translation)

    # Transform each point to the camera frame
    points_camera = [
        np.dot(rotation_matrix, np.array(point3D)) + translation_vector
        for point3D in features_3D
    ]

    return points_camera

def cam2world(points_camera, quaternion, translation):
    """
    Transform 3D points from the camera coordinate system back to the world coordinate system.

    Parameters:
    - points_camera: List of 3D points in the camera coordinate system.
    - quaternion: Quaternion defining the camera's orientation.
    - translation: Translation vector defining the camera's position.

    Returns:
    - points_world: List of 3D points in the world coordinate system.
    """
    # Convert quaternion to a rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    rotation_matrix_inv = rotation_matrix.T  # Transpose is the inverse for rotation matrices
    translation_vector = np.array(translation)

    # Transform each point to the world frame
    points_world = [
        np.dot(rotation_matrix_inv, np.array(point_cam) - translation_vector)
        for point_cam in points_camera
    ]

    return points_world


def load_depth_map(file_path):
    """
    Loads a 16-bit depth map from a PNG file and converts it into a NumPy array.

    Parameters:
        file_path (str): Path to the depth map PNG file.

    Returns:
        np.ndarray: The depth map as a NumPy array of float values.
    """
    # Read the RGB depth map
    depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if depth_map is None:
        raise FileNotFoundError(f"Could not load the depth map from {file_path}")

    # Convert to grayscale (single channel)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Optionally scale the depth values to metric values, if a scaling factor is known
    # scale_factor = 0.001  # Example: Convert millimeters to meters
    # depth_map *= scale_factor

    return depth_map

def match_features_with_depth_map(metric_depths, depth_map):
    """
    Matches metric depths with depth map values based on pixel coordinates.

    Parameters:
    - metric_depths (list): List of tuples (x, y, metric_depth), where
        - x, y are the pixel coordinates in the image.
        - metric_depth is the depth value from COLMAP.
    - depth_map (np.ndarray): Loaded depth map as a 2D NumPy array.

    Returns:
    - colmap_depths (list): List of metric depths from COLMAP.
    - depth_estimates (list): List of corresponding depth values from the depth map.
    """
    colmap_depths = []
    depth_estimates = []

    for x, y, metric_depth in metric_depths:
        px, py = int(x), int(y)  # Ensure pixel indices are integers

        # Ensure the pixel is within bounds
        if 0 <= px < depth_map.shape[1] and 0 <= py < depth_map.shape[0]:
            depth_map_value = depth_map[py, px]  # (y, x) indexing for images
            if depth_map_value > 0:  # Valid depth value
                colmap_depths.append(metric_depth)
                depth_estimates.append(depth_map_value)

    return colmap_depths, depth_estimates



def match_depths_with_map(metric_depths, depth_map):
    """
    Matches metric depths with corresponding depth map values.
    """
    matches = []
    for x, y, metric_depth in metric_depths:
        px, py = int(x), int(y)
        if 0 <= px < depth_map.shape[1] and 0 <= py < depth_map.shape[0]:
            depth_map_value = depth_map[py, px]
            if depth_map_value > 0:  # Valid depth map value
                matches.append((metric_depth, depth_map_value))
    return matches

def match_and_align_depths(spherical_with_depth, depth_map, equirectangular_converter):
    """
    Match metric depths with estimated depths, and align them using least squares.
    
    Parameters:
    - spherical_with_depth (list): List of features with spherical coordinates and depths.
    - depth_map (np.ndarray): Panoramic depth map as a 2D array.
    - equirectangular_converter (EquirectangularToSpherical): Instance for spherical-pixel conversions.
    
    Returns:
    - aligned_depth_map (np.ndarray): Depth map adjusted with the computed scale and bias.
    - scale (float): Optimal scale factor.
    - bias (float): Optimal bias value.
    """
    matched_depths = []

    for feature in spherical_with_depth:
        phi, theta = feature["phi"], feature["theta"]
        metric_depth = feature["depth"]

        # Convert spherical coordinates to pixel coordinates
        x, y = equirectangular_converter.spherical_to_pixel(phi, theta)

        # Ensure the coordinates are within the depth map bounds
        if 0 <= x < equirectangular_converter._width and 0 <= y < equirectangular_converter._height:
            depth_estimate = depth_map[y, x]
            matched_depths.append((metric_depth, depth_estimate))

    # Separate metric depths and depth estimates for alignment
    colmap_depths, depth_estimates = zip(*matched_depths)
    colmap_depths = np.array(colmap_depths)
    depth_estimates = np.array(depth_estimates)

    # Align using least squares to find scale and bias
    scale, bias = align_depths(colmap_depths, depth_estimates)

    # Apply the scale and bias to the entire depth map
    aligned_depth_map = apply_scale_bias(depth_map, scale, bias)
    return aligned_depth_map, scale, bias

def pixel_to_spherical(x, y, width, height, equ_cx, equ_cy):
    """
    Convert a pixel (x, y) in an equirectangular image to spherical coordinates (phi, theta).

    Parameters:
    - x (float): x-coordinate of the pixel.
    - y (float): y-coordinate of the pixel.
    - width (int): Width of the equirectangular image.
    - height (int): Height of the equirectangular image.
    - equ_cx (float): X-coordinate of the image center.
    - equ_cy (float): Y-coordinate of the image center.

    Returns:
    - phi (float): Longitude in degrees, ranging from -180 to 180.
    - theta (float): Latitude in degrees, ranging from -90 to 90.
    """
    phi = (x - equ_cx) / equ_cx * 180
    theta = -(y - equ_cy) / equ_cy * 90
    return phi, theta



def spherical_to_cartesian(phi, theta, depth):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters:
    - phi (float): Longitude in degrees.
    - theta (float): Latitude in degrees.
    - depth (float): Depth value (distance from the origin).

    Returns:
    - (X, Y, Z): Cartesian coordinates.
    """
    # Convert angles from degrees to radians
    phi_rad = np.radians(phi)
    theta_rad = np.radians(theta)

    # Calculate Cartesian coordinates
    X = depth * np.cos(theta_rad) * np.sin(phi_rad)
    Y = depth * np.sin(theta_rad)
    Z = depth * np.cos(theta_rad) * np.cos(phi_rad)

    return X, Y, Z

def generate_points3D_file(
    aligned_depth_map,
    original_features,
    rgb_image,
    output_file,
    start_id,
    error_value,
    image_id,
    quaternion,
    translation,
    start_point2d_idx,
):
    """
    Generate a new points3D.txt file with additional feature points from the aligned depth map.

    Parameters:
    - aligned_depth_map (np.ndarray): 2D depth map after alignment.
    - original_features (list): List of original feature points as (x, y).
    - rgb_image (np.ndarray): Original RGB image corresponding to the depth map.
    - output_file (str): Path to the output points3D.txt file.
    - start_id (int): Starting ID for new POINT3D_IDs.
    - error_value (float): Reprojection error for all new points.
    - image_id (int): ID of the image these features belong to.
    - quaternion (list): Quaternion defining the camera's orientation.
    - translation (list): Translation vector defining the camera's position.
    - start_point2d_idx (int): Starting index for POINT2D_IDX.

    Returns:
    - new_points2d (list): List of (x, y, POINT3D_ID, POINT2D_IDX) for images.txt.
    """
    # Track original feature points
    original_feature_set = set((int(x), int(y)) for x, y in original_features)

    # Prepare the new points
    height, width = aligned_depth_map.shape
    equ_cx, equ_cy = width / 2, height / 2
    new_points = []
    new_points2d = []  # To store (x, y, POINT3D_ID, POINT2D_IDX) for images.txt

    point3d_id = start_id
    point2d_idx = start_point2d_idx  # Initialize POINT2D_IDX

    # Use a dictionary to track unique 3D points
    unique_points = {}

    for y in range(height):
        for x in range(width):
            # Skip pixels that are already in the original features
            if (x, y) in original_feature_set:
                continue

            # Get the depth value
            depth = aligned_depth_map[y, x]
            if depth <= 0:  # Skip invalid or zero depth
                continue

            # Convert pixel to spherical coordinates
            phi, theta = pixel_to_spherical(x, y, width, height, equ_cx, equ_cy)

            # Convert spherical to Cartesian coordinates in the camera frame
            X_cam, Y_cam, Z_cam = spherical_to_cartesian(phi, theta, depth)

            # Create a unique hashable key for the 3D point
            point_key = (round(X_cam, 6), round(Y_cam, 6), round(Z_cam, 6))

            if point_key in unique_points:
                continue  # Skip duplicate points

            # Add to unique points
            unique_points[point_key] = point3d_id

            # Get RGB color from the original image
            r, g, b = rgb_image[y, x]

            # Add this point to the list
            new_points.append({
                "point3d_id": point3d_id,
                "xyz_camera": (X_cam, Y_cam, Z_cam),
                "rgb": (r, g, b),
                "error": error_value,
                "track": (image_id, point2d_idx)  # TRACK[] with IMAGE_ID and POINT2D_IDX
            })

            # Add to new_points2d for `images.txt`
            new_points2d.append((x, y, point3d_id, point2d_idx))

            # Increment IDs
            point3d_id += 1
            point2d_idx += 1

    # Transform points from camera to world coordinates
    points_world = cam2world([point["xyz_camera"] for point in new_points], quaternion, translation)

    # Update the world coordinates in the new points
    for i, point in enumerate(new_points):
        point["xyz_world"] = points_world[i]

    # Write the new points to the file
    with open(output_file, 'w') as f:
        # Write the header
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(new_points)}\n")
        
        # Write the points with higher precision
        for point in new_points:
            f.write(f"{point['point3d_id']} {point['xyz_world'][0]:.17f} {point['xyz_world'][1]:.17f} {point['xyz_world'][2]:.17f} "
                    f"{point['rgb'][0]} {point['rgb'][1]} {point['rgb'][2]} {point['error']:.17f} "
                    f"{point['track'][0]} {point['track'][1]}\n")

    print(f"New points3D.txt file created at: {output_file}")
    return new_points2d

def generate_new_points_file_single_line(new_points2d, output_file):
    """
    Create a new file containing POINTS2D[] as (X, Y, POINT3D_ID) on a single line.

    Parameters:
    - new_points2d (list): List of (x, y, POINT3D_ID, POINT2D_IDX).
    - output_file (str): Path to the output file.

    Returns:
    - None
    """
    with open(output_file, 'w') as f:
        # Write a header for clarity
        f.write("# New points file containing POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Format: X Y POINT3D_ID\n")

        # Create a single line string for all entries
        points_line = " ".join(f"{x:.2f} {y:.2f} {point3d_id}" for x, y, point3d_id, _ in new_points2d)
        
        # Write the points to the file on a single line
        f.write(points_line + "\n")

    print(f"New points file created at: {output_file}")
