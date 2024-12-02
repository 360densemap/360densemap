import os
from PIL import Image

def shrink_images_in_repo(repo_path, output_path=None, resize_factor=0.5):
    """
    Shrinks all images in the specified repository by the given resize factor.

    Parameters:
    - repo_path: str, path to the repository containing images.
    - output_path: str, path to save the resized images (default: None, overwrites original images).
    - resize_factor: float, the factor by which to resize images (default: 0.25 for 1/4 size).
    """
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    # Supported image formats
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                input_file_path = os.path.join(root, file)
                try:
                    # Open the image
                    with Image.open(input_file_path) as img:
                        # Calculate new dimensions
                        new_width = int(img.width * resize_factor)
                        new_height = int(img.height * resize_factor)

                        # Resize the image
                        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

                        # Save the resized image
                        if output_path:
                            # Maintain directory structure in output path
                            relative_path = os.path.relpath(root, repo_path)
                            target_dir = os.path.join(output_path, relative_path)
                            os.makedirs(target_dir, exist_ok=True)
                            output_file_path = os.path.join(target_dir, file)
                        else:
                            output_file_path = input_file_path

                        resized_img.save(output_file_path)
                        print(f"Resized image saved to: {output_file_path}")
                except Exception as e:
                    print(f"Failed to process {input_file_path}: {e}")

# Example usage:
# Change 'repo_path' to the path of your repository.
repo_path = "./supercolmap_panorama/masks"
output_path = "./supercolmap_panorama/masks_small"  # Set to None to overwrite original images
shrink_images_in_repo(repo_path, output_path)