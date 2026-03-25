"""
Comparative Filling Methods for Research Benchmarking

Provides alternative hole-filling strategies to compare against Cubic Bezier:
  1. Linear Interpolation (simplest baseline)
  2. B-Spline Interpolation (scipy-based parametric curve)

Each method follows the same interface as cubic_bezier_fill_gap() so they
can be swapped in directly.
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial import KDTree
import time


# =============================================================================
# 1. LINEAR INTERPOLATION (BASELINE)
# =============================================================================

def linear_fill_gap(p_left, p_right, num_points=20, **kwargs):
    """
    Fill gap with simple linear interpolation between two boundary points.

    This is the simplest possible baseline — a straight line.
    """
    p_left = np.asarray(p_left, dtype=np.float64)
    p_right = np.asarray(p_right, dtype=np.float64)

    t_vals = np.linspace(0.0, 1.0, num_points)
    curve_pts = np.outer(1.0 - t_vals, p_left) + np.outer(t_vals, p_right)

    return curve_pts


# =============================================================================
# 2. B-SPLINE INTERPOLATION
# =============================================================================

def bspline_fill_gap(p_left, p_right, num_points=20,
                     neighbors_left=None, neighbors_right=None, **kwargs):
    """
    Fill gap using a cubic B-Spline that passes through boundary points
    and is guided by neighbour information.

    Uses scipy's make_interp_spline with 4 control knots:
      - p_left
      - midpoint shifted by left-neighbour tangent
      - midpoint shifted by right-neighbour tangent
      - p_right
    """
    p_left = np.asarray(p_left, dtype=np.float64)
    p_right = np.asarray(p_right, dtype=np.float64)

    # Build intermediate control points from neighbours
    mid = (p_left + p_right) / 2.0

    if neighbors_left is not None and len(neighbors_left) > 0:
        nl = np.asarray(neighbors_left)
        offset_l = np.mean(nl - p_left, axis=0) * 0.3
    else:
        offset_l = np.zeros(3)

    if neighbors_right is not None and len(neighbors_right) > 0:
        nr = np.asarray(neighbors_right)
        offset_r = np.mean(nr - p_right, axis=0) * 0.3
    else:
        offset_r = np.zeros(3)

    # 4 data points for spline interpolation
    ctrl_pts = np.array([
        p_left,
        p_left + (p_right - p_left) * 0.33 + offset_l,
        p_left + (p_right - p_left) * 0.67 + offset_r,
        p_right,
    ])
    knot_t = np.array([0.0, 0.33, 0.67, 1.0])

    try:
        spline = make_interp_spline(knot_t, ctrl_pts, k=3)
        t_eval = np.linspace(0.0, 1.0, num_points)
        curve_pts = spline(t_eval)
    except Exception:
        # Fallback to linear if spline fails
        curve_pts = linear_fill_gap(p_left, p_right, num_points)

    return curve_pts


# =============================================================================
# 3. NEAREST NEIGHBOR INTERPOLATION
# =============================================================================

def nearest_neighbor_fill_gap(p_left, p_right, num_points=20,
                              neighbors_left=None, neighbors_right=None, **kwargs):
    """
    Fill gap by sampling points along the line and projecting each
    onto the nearest neighbour direction. A simple geometric baseline.
    """
    p_left = np.asarray(p_left, dtype=np.float64)
    p_right = np.asarray(p_right, dtype=np.float64)

    # Start with linear
    t_vals = np.linspace(0.0, 1.0, num_points)
    curve_pts = np.outer(1.0 - t_vals, p_left) + np.outer(t_vals, p_right)

    # Pull towards neighbours if available
    all_nb = []
    if neighbors_left is not None and len(neighbors_left) > 0:
        all_nb.append(np.asarray(neighbors_left))
    if neighbors_right is not None and len(neighbors_right) > 0:
        all_nb.append(np.asarray(neighbors_right))

    if all_nb:
        nb_pts = np.vstack(all_nb)
        tree = KDTree(nb_pts)
        for i in range(len(curve_pts)):
            dist, idx = tree.query(curve_pts[i])
            if dist > 1e-10:
                pull = (nb_pts[idx] - curve_pts[i]) * 0.15
                curve_pts[i] += pull

    return curve_pts


# =============================================================================
# FILL-ALL WRAPPERS (same interface as hole_filler.fill_all_gaps)
# =============================================================================

FILL_METHODS = {
    'linear': linear_fill_gap,
    'bspline': bspline_fill_gap,
    'nearest': nearest_neighbor_fill_gap,
}


def fill_all_gaps_with_method(points, gap_pairs, method='linear',
                              num_points_per_gap=20, neighbor_k=5):
    """
    Fill all detected gap pairs with a specified method.

    Parameters
    ----------
    points     : array (N, 3)
    gap_pairs  : list of (p_left, p_right, dist, axis_val)
    method     : 'linear' | 'bspline' | 'nearest'
    num_points_per_gap : int
    neighbor_k : int

    Returns
    -------
    filled_points : ndarray (M, 3)
    """
    fill_fn = FILL_METHODS.get(method, linear_fill_gap)

    pts_array = np.asarray(points)
    tree = KDTree(pts_array)
    filled = []

    for p_left, p_right, dist, axis_val in gap_pairs:
        _, idx_left = tree.query(p_left, k=min(neighbor_k + 1, len(pts_array)))
        _, idx_right = tree.query(p_right, k=min(neighbor_k + 1, len(pts_array)))

        nb_left = pts_array[[i for i in idx_left if not np.allclose(pts_array[i], p_left)]]
        nb_right = pts_array[[i for i in idx_right if not np.allclose(pts_array[i], p_right)]]

        curve = fill_fn(p_left, p_right, num_points=num_points_per_gap,
                        neighbors_left=nb_left, neighbors_right=nb_right)
        filled.append(curve)

    if not filled:
        return np.empty((0, 3))

    return np.vstack(filled)


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

def run_comparison(points, gap_pairs, methods=None,
                   num_points_per_gap=20, neighbor_k=5):
    """
    Run multiple filling methods on the same gap data and return results
    with timing for each.

    Parameters
    ----------
    points    : array (N, 3) — cleaned point cloud
    gap_pairs : list of gap tuples
    methods   : list of method names (default: all available)

    Returns
    -------
    dict of method_name -> {
        'filled_points': ndarray,
        'time_seconds': float,
        'num_points': int,
    }
    """
    if methods is None:
        methods = list(FILL_METHODS.keys())

    # Also include bezier from hole_filler
    from hole_filler import fill_all_gaps as bezier_fill

    results = {}

    # Bezier (our method)
    t0 = time.perf_counter()
    bezier_pts = bezier_fill(points, gap_pairs,
                             num_points_per_gap=num_points_per_gap,
                             neighbor_k=neighbor_k)
    t_bezier = time.perf_counter() - t0
    results['bezier'] = {
        'filled_points': bezier_pts,
        'time_seconds': t_bezier,
        'num_points': len(bezier_pts),
    }

    # Other methods
    for method_name in methods:
        t0 = time.perf_counter()
        filled = fill_all_gaps_with_method(
            points, gap_pairs, method=method_name,
            num_points_per_gap=num_points_per_gap,
            neighbor_k=neighbor_k
        )
        elapsed = time.perf_counter() - t0
        results[method_name] = {
            'filled_points': filled,
            'time_seconds': elapsed,
            'num_points': len(filled),
        }

    return results
