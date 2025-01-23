import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

import torch
from . import kalman_filter
import time
        
def group_and_sum(A, B, C):
   # Example input tensors
    # A: shape (N, K), B: indices to select rows from A, C: group IDs
    selected_A = A[B]

    # Find unique group IDs and their corresponding indices
    unique_C, inverse_indices, counts = torch.unique(C, return_inverse=True, return_counts=True)

    # Initialize tensor to store summed values
    M = unique_C.shape[0]
    K = A.shape[1]
    grouped_A = torch.zeros((M, K), dtype=A.dtype)

    # Sum elements based on group ID using scatter_add_
    grouped_A.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, K), selected_A)
    
    return grouped_A, unique_C, counts


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = torch.tensor(m1, dtype=torch.long)
    m2 = torch.tensor(m2, dtype=torch.long)

    M1 = torch.sparse_coo_tensor(m1.t(), torch.ones(m1.shape[0]), (O, P)).to_dense()
    M2 = torch.sparse_coo_tensor(m2.t(), torch.ones(m2.shape[0]), (P, Q)).to_dense()

    mask = torch.matmul(M1, M2)
    match = torch.nonzero(mask, as_tuple=False).tolist()
    unmatched_O = tuple(set(range(O)) - set(i for i, j in match))
    unmatched_Q = tuple(set(range(Q)) - set(j for i, j in match))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[indices[:, 0], indices[:, 1]]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0].tolist()))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1].tolist()))

    return matches, unmatched_a, unmatched_b


def se3_distance_torch(T1, T2, alpha=1.0, beta=1.0):
    pos1, quat1 = T1[:, :3], T1[:, 3:]
    pos2, quat2 = T2[:, :3], T2[:, 3:]
    translation_dist = torch.norm(pos1 - pos2, dim=1)
    quat1 = quat1 / torch.norm(quat1, dim=1, keepdim=True)
    quat2 = quat2 / torch.norm(quat2, dim=1, keepdim=True)
    dot_product = torch.sum(quat1 * quat2, dim=1).clamp(-1.0, 1.0)
    rotation_dist = 2 * torch.acos(torch.abs(dot_product))
    total_distance = alpha * translation_dist + beta * rotation_dist
    return total_distance


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def grasp_distance(atracks, btracks):
    atgrasps = torch.tensor(atracks) if isinstance(atracks[0], list) else atracks
    btgrasps = torch.tensor(btracks) if isinstance(btracks[0], list) else btracks
    cost_matrix = se3_distance_torch(atgrasps, btgrasps)
    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    track_features = torch.tensor([t.smooth_feat for t in tracks], dtype=torch.float32)
    det_features = torch.tensor([d.curr_feat for d in detections], dtype=torch.float32)

    if metric == 'cosine':
        track_features = torch.nn.functional.normalize(track_features, dim=1)
        det_features = torch.nn.functional.normalize(det_features, dim=1)
        cost_matrix = 1 - torch.mm(track_features, det_features.t())
    else:
        cost_matrix = torch.cdist(track_features, det_features, p=2)

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_score(cost_matrix, tracks, detections):
    reid_sim = 1 - cost_matrix
    grasp_dist = grasp_distance(tracks, detections)
    grasp_sim = 1 - grasp_dist

    fuse_sim = reid_sim * (1 + grasp_sim) / 2
    det_scores = torch.tensor([d.score for d in detections]).unsqueeze(0).repeat(cost_matrix.shape[0], 1)
    fuse_cost = 1 - fuse_sim * (1 + det_scores) / 2

    return fuse_cost


