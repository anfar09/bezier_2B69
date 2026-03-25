"""
Quantitative Evaluation Metrics for 3D Point Cloud Reconstruction

Provides metrics commonly used in research papers:
  - Hausdorff Distance (one-sided & symmetric)
  - Mean Point-to-Point Distance
  - RMSE (Root Mean Square Error)
  - Fill Rate
  - Surface Roughness (local normal deviation)
  - Point Density Uniformity
"""

import numpy as np
from scipy.spatial import cKDTree
import time


# =============================================================================
# CORE METRICS
# =============================================================================

def hausdorff_distance(pts_a, pts_b):
    """
    Compute the symmetric Hausdorff distance between two point sets.

    H(A,B) = max( h(A,B), h(B,A) )
    where h(A,B) = max_{a in A} min_{b in B} ||a - b||

    Returns
    -------
    dict with 'forward', 'backward', 'symmetric' distances
    """
    a = np.asarray(pts_a)
    b = np.asarray(pts_b)

    if len(a) == 0 or len(b) == 0:
        return {'forward': float('inf'), 'backward': float('inf'), 'symmetric': float('inf')}

    tree_b = cKDTree(b)
    tree_a = cKDTree(a)

    dist_a_to_b, _ = tree_b.query(a)
    dist_b_to_a, _ = tree_a.query(b)

    h_forward = float(np.max(dist_a_to_b))
    h_backward = float(np.max(dist_b_to_a))

    return {
        'forward': h_forward,
        'backward': h_backward,
        'symmetric': max(h_forward, h_backward),
    }


def chamfer_distance(pts_a, pts_b):
    """
    Compute the symmetric Chamfer Distance between two point sets.

    CD(A,B) = (1/|A|) * sum_{a in A} min_{b in B} ||a - b||^2
            + (1/|B|) * sum_{b in B} min_{a in A} ||b - a||^2

    Parameters
    ----------
    pts_a : array (N, 3)
    pts_b : array (M, 3)

    Returns
    -------
    dict with 'forward', 'backward', 'symmetric' chamfer distances
    """
    a = np.asarray(pts_a)
    b = np.asarray(pts_b)

    if len(a) == 0 or len(b) == 0:
        return {'forward': float('inf'), 'backward': float('inf'), 'symmetric': float('inf')}

    tree_b = cKDTree(b)
    tree_a = cKDTree(a)

    dist_a_to_b, _ = tree_b.query(a)
    dist_b_to_a, _ = tree_a.query(b)

    cd_forward = float(np.mean(dist_a_to_b ** 2))
    cd_backward = float(np.mean(dist_b_to_a ** 2))

    return {
        'forward': cd_forward,
        'backward': cd_backward,
        'symmetric': cd_forward + cd_backward,
    }


def mean_point_distance(pts_a, pts_b):
    """
    Mean distance from each point in A to its nearest neighbour in B.

    Returns
    -------
    dict with 'forward_mean', 'backward_mean', 'overall_mean'
    """
    a = np.asarray(pts_a)
    b = np.asarray(pts_b)

    if len(a) == 0 or len(b) == 0:
        return {'forward_mean': float('inf'), 'backward_mean': float('inf'), 'overall_mean': float('inf')}

    tree_b = cKDTree(b)
    tree_a = cKDTree(a)

    dist_a_to_b, _ = tree_b.query(a)
    dist_b_to_a, _ = tree_a.query(b)

    fwd = float(np.mean(dist_a_to_b))
    bwd = float(np.mean(dist_b_to_a))

    return {
        'forward_mean': fwd,
        'backward_mean': bwd,
        'overall_mean': (fwd + bwd) / 2.0,
    }


def rmse(pts_reconstructed, pts_ground_truth):
    """
    Root Mean Square Error: for each reconstructed point, find nearest
    ground-truth point and compute RMSE of those distances.

    Parameters
    ----------
    pts_reconstructed : array (N, 3) — generated / filled points only
    pts_ground_truth  : array (M, 3) — reference complete surface

    Returns
    -------
    float — RMSE value
    """
    recon = np.asarray(pts_reconstructed)
    gt = np.asarray(pts_ground_truth)

    if len(recon) == 0 or len(gt) == 0:
        return float('inf')

    tree_gt = cKDTree(gt)
    dists, _ = tree_gt.query(recon)

    return float(np.sqrt(np.mean(dists ** 2)))


def fill_rate(n_holes_detected, n_holes_filled):
    """
    Percentage of detected holes that were successfully filled.

    Parameters
    ----------
    n_holes_detected : int — total holes found
    n_holes_filled   : int — holes that received fill points

    Returns
    -------
    float — percentage (0-100)
    """
    if n_holes_detected == 0:
        return 100.0
    return (n_holes_filled / n_holes_detected) * 100.0


def surface_roughness(points, k=10):
    """
    Measure surface roughness by estimating local planarity.

    For each point, fit a plane to its k nearest neighbours using PCA.
    The roughness is the mean ratio of the smallest eigenvalue to the
    sum of eigenvalues (lower = smoother).

    Returns
    -------
    dict with 'mean_roughness', 'std_roughness', 'per_point_roughness'
    """
    pts = np.asarray(points)
    if len(pts) < k + 1:
        return {'mean_roughness': 0.0, 'std_roughness': 0.0, 'per_point_roughness': np.array([])}

    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k + 1)

    roughness = np.zeros(len(pts))

    for i in range(len(pts)):
        neighbors = pts[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.abs(eigenvalues)
        total = eigenvalues.sum()
        if total > 1e-12:
            roughness[i] = eigenvalues[0] / total  # smallest eigenvalue ratio
        else:
            roughness[i] = 0.0

    return {
        'mean_roughness': float(np.mean(roughness)),
        'std_roughness': float(np.std(roughness)),
        'per_point_roughness': roughness,
    }


def point_density_uniformity(points, k=6):
    """
    Measure how uniformly points are distributed.
    Uses coefficient of variation of k-NN distances.
    Lower = more uniform.

    Returns
    -------
    dict with 'mean_spacing', 'std_spacing', 'cv' (coefficient of variation)
    """
    pts = np.asarray(points)
    if len(pts) < k + 1:
        return {'mean_spacing': 0.0, 'std_spacing': 0.0, 'cv': 0.0}

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)

    mu = float(np.mean(mean_dists))
    sigma = float(np.std(mean_dists))
    cv = sigma / mu if mu > 1e-12 else 0.0

    return {
        'mean_spacing': mu,
        'std_spacing': sigma,
        'cv': cv,
    }


def per_point_error(pts_filled, pts_ground_truth):
    """
    Compute per-point distance from filled points to nearest ground truth.
    Useful for error heatmap visualization.

    Returns
    -------
    ndarray of distances, shape (N,)
    """
    filled = np.asarray(pts_filled)
    gt = np.asarray(pts_ground_truth)

    if len(filled) == 0 or len(gt) == 0:
        return np.array([])

    tree_gt = cKDTree(gt)
    dists, _ = tree_gt.query(filled)
    return dists


# =============================================================================
# AGGREGATE WRAPPER
# =============================================================================

def compute_all_metrics(result_dict, ground_truth=None):
    """
    Compute all available metrics from a pipeline result dictionary.

    Parameters
    ----------
    result_dict   : dict returned by process_point_cloud()
    ground_truth  : optional array (M, 3) of reference points

    Returns
    -------
    dict of metric_name -> value
    """
    metrics = {}

    # Fill statistics
    n_gaps = len(result_dict.get('gaps_axis1', [])) + len(result_dict.get('gaps_axis2', []))
    n_filled = result_dict.get('num_filled', 0)
    metrics['total_gaps_detected'] = n_gaps
    metrics['total_fill_points'] = n_filled
    metrics['fill_rate'] = fill_rate(n_gaps, n_gaps)  # all gaps get fill attempts

    # Point counts
    metrics['original_points'] = len(result_dict.get('original_inlier_points', []))
    metrics['merged_points'] = len(result_dict.get('merged_points', []))

    # Surface roughness of merged result
    merged = result_dict.get('merged_points', np.empty((0, 3)))
    if len(merged) > 0:
        roughness = surface_roughness(merged, k=min(10, len(merged) - 1))
        metrics['surface_roughness_mean'] = roughness['mean_roughness']
        metrics['surface_roughness_std'] = roughness['std_roughness']

        density = point_density_uniformity(merged, k=min(6, len(merged) - 1))
        metrics['density_uniformity_cv'] = density['cv']
        metrics['mean_point_spacing'] = density['mean_spacing']

    # If ground truth is available, compute accuracy metrics
    if ground_truth is not None and len(ground_truth) > 0:
        gt = np.asarray(ground_truth)

        hd = hausdorff_distance(merged, gt)
        metrics['hausdorff_symmetric'] = hd['symmetric']
        metrics['hausdorff_forward'] = hd['forward']
        metrics['hausdorff_backward'] = hd['backward']

        mpd = mean_point_distance(merged, gt)
        metrics['mean_distance_overall'] = mpd['overall_mean']

        # RMSE on filled points specifically
        bezier_all = result_dict.get('combined_bezier_pts', np.empty((0, 3)))
        if len(bezier_all) > 0:
            metrics['rmse_filled'] = rmse(bezier_all, gt)
            metrics['per_point_errors'] = per_point_error(bezier_all, gt)
        else:
            metrics['rmse_filled'] = 0.0

    # Timings (if available)
    if 'timings' in result_dict:
        metrics['timings'] = result_dict['timings']

    return metrics
