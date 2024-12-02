import cv2
import os

def resize_images_in_folder(input_folder, output_folder):
    # Check if the output folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        # Full path for the input image
        input_image_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
            print(f"Skipping non-image file: {filename}")
            continue

        # Load the original image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error: Unable to load image at {input_image_path}")
            continue

        # Get the original dimensions
        original_height, original_width = image.shape[:2]

        # Resize the image to half its original dimensions
        resized_image = cv2.resize(image, (original_width // 2, original_height // 2))

        # Save the resized image to the output folder
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, resized_image)
        print(f"Resized image saved at {output_image_path}")

# Example usage
input_folder = './20_images_large'   # Replace with the path to your input folder
output_folder = './20_images_half'  # Replace with the path to your output folder
resize_images_in_folder(input_folder, output_folder)
