# scripts/run_alignment.py
from depth_alignment.alignment import align_depths, apply_scale_bias
from depth_alignment.utils import load_depth_map, load_colmap_depths

def main():
    # Load data
    colmap_depths = load_colmap_depths('data/colmap_depths.txt')
    depth_estimates = load_depth_map('data/depth_estimate.npy')

    # Align depths
    scale, bias = align_depths(colmap_depths, depth_estimates)
    print(f"Scale: {scale}, Bias: {bias}")

    # Apply scale and bias to an example depth map
    aligned_depth = apply_scale_bias(depth_estimates, scale, bias)
    print("Aligned depth:", aligned_depth)

if __name__ == "__main__":
    main()
