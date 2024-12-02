import torch
import numpy as np

def spherical_nn_matcher(descriptors1, descriptors2):
    matches = []
    seen_matches = set()  # Track seen matches
    for i, desc1 in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        j = np.argmin(distances)
        if distances[j] < 0.7 and (i, j) not in seen_matches:  # Adjust threshold if needed
            matches.append([i, j])
            seen_matches.add((i, j))
    return np.array(matches, dtype=np.uint32)



# Mutual nearest neighbors matcher for L2 normalized descriptors.
def mutual_nn_matcher(des1, des2):
    """
    Mutual nearest neighbor matcher for descriptors.
    Args:
        des1: torch.Tensor of shape (N1, D)
        des2: torch.Tensor of shape (N2, D)
    Returns:
        matches: numpy array of shape (M, 2) with pairs of matched indices
    """
    # Compute similarity matrix (N1 x N2)
    sim = des1 @ des2.t()  # Matrix multiplication

    # Find nearest neighbors
    nn12 = sim.max(dim=1).indices  # Best match for each descriptor in des1
    nn21 = sim.max(dim=0).indices  # Best match for each descriptor in des2

    # Mutual matching
    mutual_matches = []
    for i, j in enumerate(nn12):
        if nn21[j] == i:  # Mutual nearest neighbor check
            mutual_matches.append([i, j.item()])

    return np.array(mutual_matches)

# Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


# Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()