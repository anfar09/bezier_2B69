"""
Controlled Experiments for Research Validation

Provides tools to:
  1. Generate synthetic holes (spherical removal) in complete point clouds
  2. Add Gaussian noise at various levels
  3. Run parameter sensitivity experiments
  4. Run hole-size experiments
  5. Run noise-sensitivity experiments
"""

import numpy as np
from scipy.spatial import cKDTree
import time


# =============================================================================
# 1. SYNTHETIC HOLE GENERATION
# =============================================================================

def generate_synthetic_hole(points, center=None, radius=None):
    """
    Remove points within a sphere of given radius around a center point,
    simulating a hole in the point cloud.

    Parameters
    ----------
    points : array (N, 3)
    center : array (3,) — if None, uses centroid of a random local region
    radius : float — if None, uses 5% of the bounding box diagonal

    Returns
    -------
    dict with:
        'points_with_hole' : remaining points after removal
        'removed_points'   : the points that were removed (ground truth for hole)
        'center'           : hole center used
        'radius'           : hole radius used
        'mask'             : boolean mask (True = kept)
    """
    pts = np.asarray(points, dtype=np.float64)

    if radius is None:
        bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        radius = bbox_diag * 0.05

    if center is None:
        # Pick a random surface point as center
        idx = np.random.randint(len(pts))
        center = pts[idx].copy()

    center = np.asarray(center, dtype=np.float64)

    # Remove points within sphere
    dists = np.linalg.norm(pts - center, axis=1)
    mask = dists > radius

    return {
        'points_with_hole': pts[mask],
        'removed_points': pts[~mask],
        'center': center,
        'radius': radius,
        'mask': mask,
        'original_points': pts,
    }


def generate_random_holes(points, n_holes=3, radius_range=(None, None), seed=42):
    """
    Generate multiple random holes in the point cloud.

    Parameters
    ----------
    points       : array (N, 3)
    n_holes      : int — number of holes to generate
    radius_range : (min_radius, max_radius) — if None, auto-computed
    seed         : int — random seed for reproducibility

    Returns
    -------
    dict with:
        'points_with_holes' : remaining points
        'all_removed'       : concatenated removed points
        'holes'             : list of individual hole info dicts
        'original_points'   : the input points
    """
    rng = np.random.RandomState(seed)
    pts = np.asarray(points, dtype=np.float64)
    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))

    r_min = radius_range[0] if radius_range[0] is not None else bbox_diag * 0.03
    r_max = radius_range[1] if radius_range[1] is not None else bbox_diag * 0.08

    remaining = pts.copy()
    all_removed = []
    holes_info = []

    for i in range(n_holes):
        if len(remaining) < 10:
            break

        # Pick random center from remaining points
        idx = rng.randint(len(remaining))
        center = remaining[idx].copy()
        radius = rng.uniform(r_min, r_max)

        dists = np.linalg.norm(remaining - center, axis=1)
        mask = dists > radius
        removed = remaining[~mask]

        if len(removed) == 0:
            continue

        remaining = remaining[mask]
        all_removed.append(removed)
        holes_info.append({
            'center': center,
            'radius': radius,
            'n_removed': len(removed),
        })

    all_removed_arr = np.vstack(all_removed) if all_removed else np.empty((0, 3))

    return {
        'points_with_holes': remaining,
        'all_removed': all_removed_arr,
        'holes': holes_info,
        'original_points': pts,
        'n_holes_created': len(holes_info),
    }


# =============================================================================
# 2. NOISE INJECTION
# =============================================================================

def add_gaussian_noise(points, sigma=0.01, seed=42):
    """
    Add Gaussian noise to all points.

    Parameters
    ----------
    points : array (N, 3)
    sigma  : float — standard deviation of noise
    seed   : int — random seed

    Returns
    -------
    noisy_points : array (N, 3)
    """
    rng = np.random.RandomState(seed)
    pts = np.asarray(points, dtype=np.float64)
    noise = rng.normal(0, sigma, size=pts.shape)
    return pts + noise


# =============================================================================
# 3. EXPERIMENT RUNNERS
# =============================================================================

def run_noise_sensitivity(points, pipeline_fn, sigmas=None):
    """
    Test the pipeline's robustness to varying noise levels.

    Parameters
    ----------
    points      : array (N, 3) — clean point cloud
    pipeline_fn : callable(points) -> result_dict
    sigmas      : list of noise levels to test

    Returns
    -------
    list of dicts, each with:
        'sigma', 'result', 'time_seconds', 'num_gaps', 'num_filled'
    """
    if sigmas is None:
        # Auto-compute based on avg spacing
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        avg_spacing = float(np.mean(dists[:, 1]))
        sigmas = [0, avg_spacing * 0.1, avg_spacing * 0.5,
                  avg_spacing * 1.0, avg_spacing * 2.0]

    results = []
    for sigma in sigmas:
        if sigma > 0:
            noisy_pts = add_gaussian_noise(points, sigma=sigma)
        else:
            noisy_pts = np.asarray(points)

        t0 = time.perf_counter()
        result = pipeline_fn(noisy_pts)
        elapsed = time.perf_counter() - t0

        n_gaps = len(result.get('gaps_axis1', [])) + len(result.get('gaps_axis2', []))

        results.append({
            'sigma': sigma,
            'result': result,
            'time_seconds': elapsed,
            'num_gaps': n_gaps,
            'num_filled': result.get('num_filled', 0),
        })

    return results


def run_hole_size_experiment(points, pipeline_fn, radii=None, center=None):
    """
    Test how well the pipeline handles holes of different sizes.

    Parameters
    ----------
    points      : array (N, 3)
    pipeline_fn : callable(points) -> result_dict
    radii       : list of hole radii to test
    center      : fixed center for all holes (if None, uses centroid)

    Returns
    -------
    list of dicts, each with:
        'radius', 'n_removed', 'result', 'time_seconds'
    """
    pts = np.asarray(points, dtype=np.float64)

    if center is None:
        center = pts.mean(axis=0)

    if radii is None:
        bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        radii = [bbox_diag * r for r in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]]

    results = []
    for radius in radii:
        hole_data = generate_synthetic_hole(pts, center=center, radius=radius)
        pts_holed = hole_data['points_with_hole']
        n_removed = len(hole_data['removed_points'])

        if n_removed == 0:
            continue

        t0 = time.perf_counter()
        result = pipeline_fn(pts_holed)
        elapsed = time.perf_counter() - t0

        results.append({
            'radius': radius,
            'n_removed': n_removed,
            'result': result,
            'time_seconds': elapsed,
            'ground_truth': hole_data['removed_points'],
            'original_points': pts,
        })

    return results


def run_ablation_experiment(points, pipeline_fn_with_flags):
    """
    Run ablation study by toggling components on/off.

    Parameters
    ----------
    points : array (N, 3)
    pipeline_fn_with_flags : callable(points, use_sor, use_pca, use_cross_hatch) -> result_dict

    Returns
    -------
    dict of config_name -> result_dict
    """
    configs = {
        'Full Pipeline':        {'use_sor': True,  'use_pca': True,  'use_cross_hatch': True},
        'No SOR':               {'use_sor': False, 'use_pca': True,  'use_cross_hatch': True},
        'No PCA':               {'use_sor': True,  'use_pca': False, 'use_cross_hatch': True},
        'Single Axis (No CH)':  {'use_sor': True,  'use_pca': True,  'use_cross_hatch': False},
        'Minimal (No SOR+PCA)': {'use_sor': False, 'use_pca': False, 'use_cross_hatch': True},
    }

    results = {}
    for name, flags in configs.items():
        t0 = time.perf_counter()
        result = pipeline_fn_with_flags(points, **flags)
        elapsed = time.perf_counter() - t0
        result['ablation_time'] = elapsed
        results[name] = result

    return results
