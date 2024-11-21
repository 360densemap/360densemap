import os
import glob
import numpy as np
import sqlite3

def create_colmap_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
    cursor.execute("CREATE TABLE IF NOT EXISTS descriptors (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
    cursor.execute("CREATE TABLE IF NOT EXISTS matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER, cols INTEGER, data BLOB)")
    return conn, cursor

def save_keypoints(cursor, image_id, keypoints):
    rows, cols = keypoints.shape
    data = keypoints.tobytes()
    cursor.execute("INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)", (image_id, rows, cols, data))

def save_descriptors(cursor, image_id, descriptors):
    rows, cols = descriptors.shape
    data = descriptors.tobytes()
    cursor.execute("INSERT INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, ?, ?)", (image_id, rows, cols, data))

def save_matches(cursor, image_id1, image_id2, matches):
    rows, cols = matches.shape
    data = matches.tobytes()
    pair_id = image_id1 * (2**31 - 1) + image_id2  # Create a unique pair_id
    cursor.execute("INSERT INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)", (pair_id, rows, cols, data))

def process_superglue_output(output_dir, db_path, images_dir):
    # Initialize the database
    conn, cursor = create_colmap_database(db_path)

    # Map image names to IDs based on the format 'image_XXXX' with padding
    image_id_map = {}
    for idx, image_name in enumerate(sorted(os.listdir(images_dir))):
        base_name = os.path.splitext(image_name)[0]  # Remove file extension
        image_id = int(base_name.split('_')[-1])  # Extract numerical ID from name
        image_id_map[base_name] = image_id  # Store in map

    # Process keypoints and descriptors
    keypoint_files = sorted(glob.glob(os.path.join(output_dir, "keypoints*.npy")))
    descriptor_files = sorted(glob.glob(os.path.join(output_dir, "descriptors*.npy")))

    for i, keypoint_file in enumerate(keypoint_files):
        keypoints = np.load(keypoint_file)
        descriptors = np.load(descriptor_files[i])

        base_name = f"image_{i+1:04d}"  # Adjust to match 1-based index image naming convention
        image_id = image_id_map.get(base_name)

        if image_id is not None:
            save_keypoints(cursor, image_id, keypoints)
            save_descriptors(cursor, image_id, descriptors)
        else:
            print(f"Warning: Image ID for keypoints/descriptors {i+1} not found in image map.")

    # Process matches for image pairs
    match_files = sorted(glob.glob(os.path.join(output_dir, "matches_*_*.npy")))
    for match_file in match_files:
        match_data = np.load(match_file)

        # Extract indices from the filename, e.g., "matches_1_2.npy"
        base_name = os.path.basename(match_file).replace("matches_", "").replace(".npy", "")
        image1_idx, image2_idx = map(int, base_name.split("_"))  # Convert indices to integers

        # Get corresponding image IDs
        image_id1 = image_id_map.get(f"image_{image1_idx:04d}")
        image_id2 = image_id_map.get(f"image_{image2_idx:04d}")

        if image_id1 is not None and image_id2 is not None:
            save_matches(cursor, image_id1, image_id2, match_data)
        else:
            print(f"Warning: Match data between {image1_idx} and {image2_idx} could not be saved due to missing image IDs.")

    # Commit and close database
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process SuperGlue output and convert to COLMAP database format")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory containing SuperGlue output files")
    parser.add_argument("--db_path", required=True, type=str, help="Path to the COLMAP database file")
    parser.add_argument("--images_dir", required=True, type=str, help="Directory containing image files")
    
    args = parser.parse_args()
    process_superglue_output(args.output_dir, args.db_path, args.images_dir)
