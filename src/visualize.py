import matplotlib.pyplot as plt
import numpy as np
import cv2

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


def visualize_spherical_with_depth(spherical_with_depth):
    """
    Visualize spherical coordinates with depths as a scatter plot.

    Parameters:
    - spherical_with_depth (list): List of {"phi", "theta", "depth"} dictionaries.
    """
    phis = [feature["phi"] for feature in spherical_with_depth]
    thetas = [feature["theta"] for feature in spherical_with_depth]
    depths = [feature["depth"] for feature in spherical_with_depth]

    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(phis, thetas, c=depths, cmap='jet')
    plt.colorbar(scatter, label="Depth")
    plt.xlabel("Phi (Longitude)")
    plt.ylabel("Theta (Latitude)")
    plt.title("Spherical Coordinates with Depths")
    plt.show()
