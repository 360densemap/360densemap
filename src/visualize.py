import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils import pixel_to_spherical, spherical_to_cartesian


def visualize_3d_points(points_camera):
    """
    Visualize 3D points in the camera frame using a scatter plot.

    Parameters:
    - points_camera (list): List of (x, y, z) points in the camera coordinate system.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    xs = [point[0] for point in points_camera]
    ys = [point[1] for point in points_camera]
    zs = [point[2] for point in points_camera]

    ax.scatter(xs, ys, zs, c=zs, cmap='jet', marker='o', s=10)

    ax.set_xlabel('X (Camera Frame)')
    ax.set_ylabel('Y (Camera Frame)')
    ax.set_zlabel('Z (Depth)')
    ax.set_title('3D Points in Camera Frame')

    plt.show()

def visualize_features_on_image(image_path, features):
    """
    Visualize 2D features on the equirectangular image.

    Parameters:
    - image_path (str): Path to the equirectangular image.
    - features (list): List of (x, y) feature coordinates.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.imshow(img_rgb)
    for x, y in features:
        plt.scatter(x, y, color='red', s=10)
    plt.title("Features on Equirectangular Image")
    plt.show()



def visualize_depth_alignment(original_depth, aligned_depth):
    """
    Visualize the original and aligned depth maps, and their difference.

    Parameters:
    - original_depth (np.ndarray): Original depth map.
    - aligned_depth (np.ndarray): Aligned depth map.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_depth, cmap='jet')
    plt.colorbar(label="Depth")
    plt.title("Original Depth Map")

    plt.subplot(1, 3, 2)
    plt.imshow(aligned_depth, cmap='jet')
    plt.colorbar(label="Depth")
    plt.title("Aligned Depth Map")

    plt.subplot(1, 3, 3)
    plt.imshow(aligned_depth - original_depth, cmap='seismic')
    plt.colorbar(label="Difference")
    plt.title("Difference (Aligned - Original)")

    plt.show()


def visualize_3d_depth_map_comparison(original_depth_map, aligned_depth_map, rgb_image, width, height, equ_cx, equ_cy):
    """
    Visualizes the 3D point clouds of the original and aligned depth maps side by side.

    Parameters:
    - original_depth_map (np.ndarray): 2D depth map before alignment.
    - aligned_depth_map (np.ndarray): 2D depth map after alignment.
    - rgb_image (np.ndarray): Original RGB image corresponding to the depth maps.
    - width (int): Width of the equirectangular image.
    - height (int): Height of the equirectangular image.
    - equ_cx (float): X-coordinate of the image center.
    - equ_cy (float): Y-coordinate of the image center.
    """
    original_points_3d = []
    aligned_points_3d = []
    colors = []

    for y in range(height):
        for x in range(width):
            # Get depth values for original and aligned maps
            original_depth = original_depth_map[y, x]
            aligned_depth = aligned_depth_map[y, x]
            
            if original_depth > 0:  # Only process valid depth values
                # Convert pixel to spherical coordinates
                phi, theta = pixel_to_spherical(x, y, width, height, equ_cx, equ_cy)

                # Convert spherical to Cartesian coordinates for original depth
                X_orig, Y_orig, Z_orig = spherical_to_cartesian(phi, theta, original_depth)
                original_points_3d.append((X_orig, Y_orig, Z_orig))

                # Convert spherical to Cartesian coordinates for aligned depth
                X_aligned, Y_aligned, Z_aligned = spherical_to_cartesian(phi, theta, aligned_depth)
                aligned_points_3d.append((X_aligned, Y_aligned, Z_aligned))

                # Get RGB color from the original image
                r, g, b = rgb_image[y, x]
                colors.append((r / 255.0, g / 255.0, b / 255.0))  # Normalize to [0, 1]

    # Convert lists to NumPy arrays for plotting
    original_points_3d = np.array(original_points_3d)
    aligned_points_3d = np.array(aligned_points_3d)
    colors = np.array(colors)

    # Create 3D scatter plot
    fig = plt.figure(figsize=(15, 10))

    # Plot original depth map points
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_points_3d[:, 0], original_points_3d[:, 1], original_points_3d[:, 2], c=colors, s=0.5)
    ax1.set_title("Original Depth Map")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_axis_off()  # Hide the axes
    # Plot aligned depth map points
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(aligned_points_3d[:, 0], aligned_points_3d[:, 1], aligned_points_3d[:, 2], c=colors, s=0.5)
    ax2.set_title("Aligned Depth Map")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_axis_off()  # Hide the axes

    plt.tight_layout()
    plt.show()

    
