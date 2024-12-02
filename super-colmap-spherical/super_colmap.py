from SuperPointDetectors import get_super_points_from_scenes_return
from matchers import mutual_nn_matcher
from matchers import spherical_nn_matcher
import cv2
import os, time
import numpy as np
import argparse
import torch
from database import COLMAPDatabase
from utils import remap_to_spherical

camModelDict = {'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
                'FULL_OPENCV': 5,
                'SIMPLE_RADIAL_FISHEYE': 6,
                'RADIAL_FISHEYE': 7,
                'OPENCV_FISHEYE': 8,
                'FOV': 9,
                'THIN_PRISM_FISHEYE': 10,
                'SPHERE': 11}

def get_init_cameraparams(width, height, modelId):
    f = max(width, height) * 1.2
    cx = width / 2.0
    cy = height / 2.0
    if modelId == 11:  # Spherical model for SphereSfM compatibility
        # Spherical models typically use width, height, and a focal length-like parameter
        return np.array([1, cx, cy])  # Adjusted to provide three parameters
    elif modelId == 0:
        return np.array([f, cx, cy])
    elif modelId == 1:
        return np.array([f, f, cx, cy])
    elif modelId in [2, 6]:
        return np.array([f, cx, cy, 0.0])
    elif modelId in [3, 7]:
        return np.array([f, cx, cy, 0.0, 0.0])
    elif modelId in [4, 8]:
        return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0])
    elif modelId == 9:
        return np.array([f, f, cx, cy, 0.0])
    return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def init_cameras_database(db, images_path, cameratype, single_camera):
    print("Initializing cameras in the database...")
    images_name = []
    width = None
    height = None
    for name in sorted(os.listdir(images_path)):
        if name.endswith(('.jpg', '.png')):
            images_name.append(name)
            if width is None:
                img = cv2.imread(os.path.join(images_path, name))
                height, width = img.shape[:2]
    cameraModel = camModelDict[cameratype]
    params = get_init_cameraparams(width, height, cameraModel)
    if single_camera:
        db.add_camera(cameraModel, width, height, params, camera_id=0)
    for i, name in enumerate(images_name):
        if single_camera:
            db.add_image(name, 0, image_id=i)
            continue
        db.add_camera(cameraModel, width, height, params, camera_id=i)
        db.add_image(name, i, image_id=i)
    return images_name, width, height

def import_feature(db, images_path, mask_path,images_name):
    print("feature extraction by super points ...........................")
    sps = get_super_points_from_scenes_return(images_path, mask_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    for i, name in enumerate(images_name):
        keypoints = sps[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
            np.ones((n_keypoints, 1)).astype(np.float32), np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i, keypoints)

    return sps

def match_features(db, sps, images_name, match_list_path):
    print("Matching features by exhaustive match...")
    num_images = len(images_name)
    match_list = open(match_list_path, 'w')

    for i in range(num_images):
        for j in range(i + 1, num_images):  # Compare image i with all subsequent images
            match_list.write(f"{images_name[i]} {images_name[j]}\n")
            
            # Get descriptors for the two images
            D1 = sps[images_name[i]]['descriptors']  # Shape: (D, N1)
            D2 = sps[images_name[j]]['descriptors']  # Shape: (D, N2)
            
            # Convert descriptors to PyTorch tensors
            D1 = torch.tensor(D1, dtype=torch.float32)  # Shape: (D, N1)
            D2 = torch.tensor(D2, dtype=torch.float32)  # Shape: (D, N2)
            
            # Perform mutual nearest neighbor matching
            matches = mutual_nn_matcher(D1.t(), D2.t()).astype(np.uint32) # Transpose before passing
            db.add_matches(i, j, matches)

    match_list.close()

def operate(cmd):
    print(cmd)
    start = time.perf_counter()
    os.system(cmd)
    end = time.perf_counter()
    print(f"[{cmd}] took {end - start:.2f} seconds")

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mapper(projpath, images_path):
    database_path = os.path.join(projpath, "database.db")
    colmap_sparse_path = os.path.join(projpath, "sparse")
    makedir(colmap_sparse_path)

    mapper_cmd = f"colmap mapper --database_path {database_path} --image_path {images_path} --output_path {colmap_sparse_path}"
    operate(mapper_cmd)

def geometric_verification(database_path, match_list_path):
    print("Running geometric verification...")
    cmd = f"colmap matches_importer --database_path {database_path} --match_list_path {match_list_path} --match_type pairs"
    operate(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SuperPoint integration with SphereSfM for COLMAP')
    parser.add_argument("--projpath", required=True, type=str)
    parser.add_argument("--cameraModel", type=str, default="SPHERICAL", required=False)
    parser.add_argument("--images_path", default="rgb", type=str, required=False)
    parser.add_argument("--single_camera", action='store_true')
    parser.add_argument("--mask_path", default="masks", type=str, required=False, help="Path to the masks.")

    args = parser.parse_args()
    database_path = os.path.join(args.projpath, "database.db")
    match_list_path = os.path.join(args.projpath, "image_pairs_to_match.txt")
    if os.path.exists(database_path):
        os.remove(database_path)
    images_path = os.path.join(args.projpath, args.images_path)
    mask_path = os.path.join(args.projpath, args.mask_path)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    images_name, width, height = init_cameras_database(db, images_path, args.cameraModel, args.single_camera)
    sps = import_feature(db, images_path, mask_path,images_name)  # Pass width and height
    match_features(db, sps, images_name, match_list_path)
    db.commit()
    db.close()

    #geometric_verification(database_path, match_list_path)
    #mapper(args.projpath, images_path)
    print(f"Features successfully added to {database_path}. Matching skipped.")

