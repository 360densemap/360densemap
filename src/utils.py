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

def point3D_to_spherical(camera_features):
    spherical_features = []
    for x, y, (X, Y, Z) in camera_features:
        # Compute spherical coordinates (phi, theta) for the 3D point
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arccos(Z / r)  # Angle with respect to Z-axis
        phi = np.arctan2(Y, X)    # Angle in the XY-plane from X-axis
        spherical_features.append((phi, theta, r))
    return spherical_features

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

def world2cam(points3D, image_data, chosen_image_id):
    """
    Transform 3D points from world coordinates to the camera coordinates of a specific image.
    
    Parameters:
    - points3D: Dictionary of 3D points (from `read_points3D`).
    - image_data: Dictionary of image data (from `read_images`).
    - chosen_image_id: ID of the image to use as the reference camera.
    
    Returns:
    - points_camera: List of 3D points in the chosen image's camera coordinates.
    """
    # Get the camera extrinsic parameters for the chosen image
    chosen_image = image_data[chosen_image_id]
    quaternion = chosen_image["quaternion"]
    translation = chosen_image["translation"]
    
    # Convert the quaternion to a rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    translation_vector = np.array(translation)
    
    points_camera = []
    for point_id, point_data in points3D.items():
        # 3D point in world coordinates
        point_world = np.array(point_data["xyz"])
        
        # Transform to camera coordinates: R * X + t
        point_camera = np.dot(rotation_matrix, point_world) + translation_vector
        points_camera.append((point_id, point_camera))
    
    return points_camera


def group_images_by_pose(images, grouping_criterion="name_prefix"):
    """
    Groups images by pose based on a common identifier.

    Parameters:
    - images (dict): Dictionary of images from read_images.
    - grouping_criterion (str): Method to group images (e.g., 'name_prefix').

    Returns:
    - grouped_images (dict): Dictionary where each key is a pose ID and each value is a list of IMAGE_IDs.
    """
    grouped_images = {}

    for image_id, data in images.items():
        if grouping_criterion == "name_prefix":
            # Use the full name without extension as the pose ID
            pose_id = data['name'].split('.')[0]  # e.g., 'image_0126' from 'image_0126.jpg'
        else:
            raise ValueError(f"Unknown grouping criterion: {grouping_criterion}")

        if pose_id not in grouped_images:
            grouped_images[pose_id] = []
        grouped_images[pose_id].append(image_id)

    return grouped_images


def aggregate_features_for_pose(image_data, points3D, grouped_images, pose_id):
    """
    Aggregates unique 3D feature points seen across multiple perspective views for a single pose.

    Parameters:
    - image_data (dict): Dictionary of images from read_images.
    - points3D (dict): Dictionary of 3D points from read_points3D.
    - grouped_images (dict): Dictionary of grouped images by pose.
    - pose_id (str): The identifier of the pose to aggregate features for.

    Returns:
    - aggregated_features (list): List of unique 3D features seen from this pose.
    """
    aggregated_features = []
    unique_points = set()  # To track unique 3D points

    for image_id in grouped_images[pose_id]:
        image_features = get_features_for_camera(image_data, points3D, image_id)
        for x, y, point3D in image_features:
            point3D_id = tuple(point3D)
            if point3D_id not in unique_points:
                aggregated_features.append((x, y, point3D))
                unique_points.add(point3D_id)

    return aggregated_features

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
    
    def spherical_to_pixel(self, phi, theta):
        """
        Convert spherical coordinates (phi, theta) to pixel coordinates in the equirectangular image.

        Parameters:
        - phi (float): Longitude in degrees, ranging from -180 to 180.
        - theta (float): Latitude in degrees, ranging from -90 to 90.

        Returns:
        - (x, y): Pixel coordinates in the equirectangular image.
        """
        x = int((phi / 180) * self._equ_cx + self._equ_cx)
        y = int((-theta / 90) * self._equ_cy + self._equ_cy)
        return x, y