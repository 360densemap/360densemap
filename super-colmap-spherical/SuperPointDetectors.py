from superpoint import SuperPoint
import cv2
import numpy as np
import torch
from utils import *
import json
import argparse
from tqdm import tqdm
import os

class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": "superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        print("SuperPoint detector config: ")
        print(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        print("creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image, mask=None):
        # Convert to grayscale if the input image is RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert image to a tensor for SuperPoint
        image_tensor = image2tensor(image, self.device)
        pred = self.superpoint({'image': image_tensor})

        # Extract initial keypoints, scores, and descriptors
        keypoints = pred["keypoints"][0].cpu().detach().numpy()
        scores = pred["scores"][0].cpu().detach().numpy()
        descriptors = pred["descriptors"][0].cpu().detach().numpy()

        # If a mask is provided, filter keypoints based on the mask
        if mask is not None:
            # Ensure mask is binary
            if mask.ndim == 3:  # Convert to grayscale if needed
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = (mask > 0).astype(np.uint8)  # Binary mask: 1 for valid regions, 0 for black regions
            
            # Filter keypoints based on the mask
            valid_indices = []
            for i, kp in enumerate(keypoints):
                x, y = int(kp[0]), int(kp[1])  # Keypoint positions (float to int)
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:  # Check bounds and mask value
                    valid_indices.append(i)

            # Apply filtering
            keypoints = keypoints[valid_indices]
            scores = scores[valid_indices]
            descriptors = descriptors[:, valid_indices]  # Filter descriptors based on valid indices

        # Return the filtered results
        ret_dict = {
            "image_size": [image.shape[0], image.shape[1]],
            "keypoints": keypoints,
            "scores": scores,
            "descriptors": descriptors
        }
        return ret_dict


def get_super_points_from_scenes(image_path, result_dir):
    image_names = []
    for name in os.listdir(image_path):
        if 'jpg' in name or 'png' in name:
            image_names.append(name)
    spd = SuperPointDetector()
    for name in tqdm(sorted(image_names)):
        image_name = os.path.join(image_path, name)
        ret_dict = spd(cv2.imread(image_name))
        with open(os.path.join(result_dir, name + ".json"), 'w') as f:
            json.dump(ret_dict, f)

def get_super_points_from_scenes_return(image_path, mask_path):
    image_names = []
    for name in os.listdir(image_path):
        if 'jpg' in name or 'png' in name:
            image_names.append(name)

    spd = SuperPointDetector()
    sps = {}

    for name in tqdm(sorted(image_names)):
        image_name = os.path.join(image_path, name)
        # Construct the mask name by appending ".png" to the image name
        mask_name = os.path.join(mask_path, f"{name}.png")

        # Read the image
        image = cv2.imread(image_name)

        # Read the mask if it exists
        if os.path.exists(mask_name):
            mask = cv2.imread(mask_name)
        else:
            print(f"Mask not found for {name}. Skipping masked filtering.")
            mask = None

        # Detect keypoints and descriptors, with mask filtering if available
        ret_dict = spd(image, mask=mask)
        sps[name] = ret_dict

    return sps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='super points detector')
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=False, default="../superpoints", help="real result_file = args.image_path + args.result_dir")
    args = parser.parse_args()
    result_dir = os.path.join(args.image_path, args.result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    get_super_points_from_scenes(args.image_path, result_dir)
