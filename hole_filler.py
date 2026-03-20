"""
3D Point Cloud Hole Filling via Dual-Axis Slicing & Bezier Cross-Hatching

Algorithm Pipeline:
  1. Statistical Outlier Removal (SOR)
  2. Axis Selection via Variance Analysis
  3. Dual-Axis Slicing & 1D Gap Detection
  4. Slice-by-Slice Cubic Bezier Curve Filling
  5. Cross-Hatching Surface Generation (merge both axes)
"""

import numpy as np
from scipy.spatial import cKDTree, KDTree
from collections import defaultdict


# =============================================================================
# 1. STATISTICAL OUTLIER REMOVAL
# =============================================================================

def statistical_outlier_removal(points, k=10, std_multiplier=2.0):
    """
    Remove isolated noise points using Statistical Outlier Removal.

    For each point, compute mean distance to its k-nearest neighbours.
    Points whose mean distance exceeds (global_mean + std_multiplier * global_std)
    are considered outliers.

    Parameters
    ----------
    points : array-like, shape (N, 3)
    k      : int, number of neighbours (default 10)
    std_multiplier : float, threshold multiplier (default 2.0)

    Returns
    -------
    mask : ndarray of bool, True = inlier
    """
    pts = np.asarray(points)
    if pts.shape[0] == 0:
        return np.array([], dtype=bool)

    tree = KDTree(pts)
    distances, _ = tree.query(pts, k=k + 1)   # includes self at index 0
    mean_dist = distances[:, 1:].mean(axis=1)  # exclude self

    global_mean = mean_dist.mean()
    global_std  = mean_dist.std()
    threshold   = global_mean + std_multiplier * global_std

    mask = mean_dist <= threshold
    return mask


# =============================================================================
# 2. AXIS SELECTION – VARIANCE ANALYSIS
# =============================================================================

def compute_variance_along_axes(points):
    """Return variance for each axis sorted descending."""
    pts = np.asarray(points)
    variances = {
        'X': float(np.var(pts[:, 0])),
        'Y': float(np.var(pts[:, 1])),
        'Z': float(np.var(pts[:, 2])),
    }
    # Sort axes by variance descending
    sorted_axes = sorted(variances, key=variances.get, reverse=True)
    axis_1 = sorted_axes[0]   # Primary (highest variance)
    axis_2 = sorted_axes[1]   # Secondary
    axis_3 = sorted_axes[2]   # Thickness (lowest variance)
    return axis_1, axis_2, axis_3, variances


AXIS_TO_COL = {'X': 0, 'Y': 1, 'Z': 2}


def axis_col(axis_name):
    return AXIS_TO_COL[axis_name]


# =============================================================================
# 3. DUAL-AXIS SLICING & 1D GAP DETECTION
# =============================================================================

def bin_points_along_axis(points, slice_axis, slice_thickness):
    """
    Bin points into slices along slice_axis.
    Each slice contains the 2D projection onto the remaining two axes.

    Parameters
    ----------
    points       : array (N, 3)
    slice_axis   : 'X' | 'Y' | 'Z'
    slice_thickness : float

    Returns
    -------
    slices : list of dicts, each with
             'axis_val'  : midpoint of the slice along slice_axis
             'slice_min' : slice start
             'slice_max' : slice end
             'pts_2d'    : (M, 2) projected points in the remaining 2 axes
             'pts_3d'    : (M, 3) original 3D points
    """
    pts = np.asarray(points)
    col = axis_col(slice_axis)
    vals = pts[:, col]
    mn, mx = vals.min(), vals.max()

    slices = []
    cur = mn
    while cur < mx:
        nxt = min(cur + slice_thickness, mx)
        mask = (vals >= cur) & (vals < nxt)
        slice_pts = pts[mask]

        # Project onto the two non-slice axes
        other_cols = [c for c in (0, 1, 2) if c != col]
        pts_2d = slice_pts[:, other_cols]
        slices.append({
            'axis_val': (cur + nxt) / 2.0,
            'slice_min': cur,
            'slice_max': nxt,
            'pts_2d': pts_2d,
            'pts_3d': slice_pts,
        })
        cur += slice_thickness
    return slices


def detect_gaps_in_slice(slice_data, gap_threshold):
    """
    Detect 1D point-to-point gaps within a single slice.

    Sorts points along the first projected axis, computes adjacent distances,
    and flags pairs where dist > gap_threshold as boundary pairs.

    Parameters
    ----------
    slice_data   : dict from bin_points_along_axis
    gap_threshold : float

    Returns
    -------
    gap_pairs_2d : list of (p1_2d, p2_2d, dist) tuples for the projected space
    gap_pairs_3d : list of (p1_3d, p2_3d, dist) tuples in original 3D space
    """
    pts_2d = slice_data['pts_2d']
    pts_3d = slice_data['pts_3d']

    if len(pts_2d) < 2:
        return [], []

    # Sort by first projected axis
    order = np.argsort(pts_2d[:, 0])
    pts_2d_sorted = pts_2d[order]
    pts_3d_sorted = pts_3d[order]

    gap_pairs_2d = []
    gap_pairs_3d = []

    for i in range(len(pts_2d_sorted) - 1):
        p1_2d, p2_2d = pts_2d_sorted[i], pts_2d_sorted[i + 1]
        p1_3d, p2_3d = pts_3d_sorted[i], pts_3d_sorted[i + 1]
        dist = p2_2d[0] - p1_2d[0]   # 1D distance along the sorting axis

        if dist > gap_threshold:
            gap_pairs_2d.append((p1_2d.copy(), p2_2d.copy(), float(dist)))
            gap_pairs_3d.append((p1_3d.copy(), p2_3d.copy(), float(dist)))

    return gap_pairs_2d, gap_pairs_3d


def process_axis(points, primary_axis, slice_thickness, gap_threshold):
    """
    Run full gap detection pipeline for one axis.

    Returns
    -------
    all_gap_pairs_3d : list of (p_left, p_right, dist, axis_val) per slice
    """
    slices = bin_points_along_axis(points, primary_axis, slice_thickness)
    all_gap_pairs_3d = []

    for sl in slices:
        _, gap_pairs_3d = detect_gaps_in_slice(sl, gap_threshold)
        for p_left, p_right, dist in gap_pairs_3d:
            all_gap_pairs_3d.append((p_left, p_right, dist, sl['axis_val']))

    return all_gap_pairs_3d


# =============================================================================
# 4. CUBIC BEZIER CURVE FILLING
# =============================================================================

def compute_tangent_from_neighbors(point, neighbors, strength=0.5):
    """
    Estimate surface tangent at `point` from its neighbours.
    Tangent is a weighted average of vectors from point to neighbours,
    weighted by 1/distance.  Strength scales the magnitude.
    """
    if len(neighbors) == 0:
        return np.zeros(3)

    vectors = neighbors - point          # (N, 3)
    dists   = np.linalg.norm(vectors, axis=1)
    dists[dists < 1e-10] = 1e-10           # avoid div-by-zero
    weights = 1.0 / dists
    weights /= weights.sum()

    tangent = np.dot(weights, vectors) * strength
    return tangent


def cubic_bezier_fill_gap(p_left, p_right, num_points=20, neighbors_left=None, neighbors_right=None):
    """
    Fill a gap between p_left and p_right with a cubic Bezier curve.

    Control points:
        P0 = p_left
        P1 = p_left + tangent_left
        P2 = p_right - tangent_right
        P3 = p_right

    Parameters
    ----------
    p_left, p_right : array-like (3,)
    num_points      : int, number of interpolated points
    neighbors_left  : array (N, 3), local neighbourhood for tangent estimation
    neighbors_right : array (N, 3), local neighbourhood for tangent estimation

    Returns
    -------
    curve_pts : ndarray (num_points, 3)
    """
    p_left  = np.asarray(p_left,  dtype=np.float64)
    p_right = np.asarray(p_right, dtype=np.float64)

    # Compute tangents
    t_left  = compute_tangent_from_neighbors(p_left, neighbors_left  if neighbors_left  is not None else np.empty((0,3)))
    t_right = compute_tangent_from_neighbors(p_right, neighbors_right if neighbors_right is not None else np.empty((0,3)))

    # Control points
    P0 = p_left
    P1 = p_left + t_left
    P2 = p_right - t_right
    P3 = p_right

    # Bernstein polynomials B(t) = (1-t)^3 .. (1-t)^2*t .. (1-t)*t^2 .. t^3
    t_vals = np.linspace(0.0, 1.0, num_points)
    curve_pts = np.zeros((num_points, 3), dtype=np.float64)

    for i, t in enumerate(t_vals):
        u = 1.0 - t
        b0 = u**3
        b1 = 3 * u**2 * t
        b2 = 3 * u * t**2
        b3 = t**3
        curve_pts[i] = b0*P0 + b1*P1 + b2*P2 + b3*P3

    return curve_pts


# =============================================================================
# 5. CROSS-HATCHING SURFACE GENERATION
# =============================================================================

def merge_and_average_points(original, new_points, merge_distance=1e-4):
    """
    Combine original + generated points, averaging near-duplicates.
    Uses KDTree to merge points within merge_distance.
    """
    if len(new_points) == 0:
        return np.asarray(original)

    all_pts = np.vstack([np.asarray(original), np.asarray(new_points)])
    tree    = KDTree(all_pts)

    merged = []
    used   = set()
    for i in range(len(all_pts)):
        if i in used:
            continue
        indices = tree.query_ball_point(all_pts[i], merge_distance)
        used.update(indices)
        merged.append(all_pts[list(indices)].mean(axis=0))

    return np.array(merged)


def fill_all_gaps(points, gap_pairs, num_points_per_gap=20, neighbor_k=5):
    """
    Fill all detected gap pairs with Bezier curves.

    Parameters
    ----------
    points           : array (N, 3), used only for neighbour lookups
    gap_pairs        : list of (p_left, p_right, dist, axis_val)
    num_points_per_gap : int
    neighbor_k       : int

    Returns
    -------
    filled_points : ndarray (M, 3)
    """
    pts_array = np.asarray(points)
    tree = KDTree(pts_array)
    filled = []

    for p_left, p_right, dist, axis_val in gap_pairs:
        # Find neighbours for tangent estimation
        _, idx_left  = tree.query(p_left,  k=min(neighbor_k + 1, len(pts_array)))
        _, idx_right = tree.query(p_right, k=min(neighbor_k + 1, len(pts_array)))

        # Exclude the boundary points themselves from neighbour set
        nb_left  = pts_array[[i for i in idx_left  if not np.allclose(pts_array[i], p_left)]]
        nb_right = pts_array[[i for i in idx_right if not np.allclose(pts_array[i], p_right)]]

        curve = cubic_bezier_fill_gap(p_left, p_right, num_points=num_points_per_gap,
                                       neighbors_left=nb_left, neighbors_right=nb_right)
        filled.append(curve)

    if not filled:
        return np.empty((0, 3))

    return np.vstack(filled)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_point_cloud(points,
                        slice_thickness=None,
                        gap_threshold=None,
                        num_points_per_gap=20,
                        neighbor_k=5,
                        sor_k=10,
                        sor_std_multiplier=2.0,
                        verbose=True):
    """
    Complete hole-filling pipeline.

    Parameters
    ----------
    points                : array-like (N, 3)
    slice_thickness       : float, bin width along slicing axes.
                             If None, auto-computed as avg_point_spacing × 2.
    gap_threshold         : float, 1D distance to flag as a gap.
                             If None, auto-computed as avg_point_spacing × 5.
    num_points_per_gap    : int, Bezier interpolation resolution
    neighbor_k            : int, neighbours for tangent estimation
    sor_k                 : int, SOR neighbour count
    sor_std_multiplier    : float, SOR threshold
    verbose               : bool

    Returns
    -------
    result : dict with keys:
        'original_inlier_points'    : ndarray (M, 3), SOR-filtered inlier points
        'bezier_pts_axis1'          : ndarray (P, 3), Bezier fill from primary axis
        'bezier_pts_axis2'          : ndarray (Q, 3), Bezier fill from secondary axis
        'combined_bezier_pts'       : ndarray (P+Q, 3), vstacked axis1+axis2 fill points
        'merged_points'             : ndarray (J, 3), merged (avg of near-duplicates)
        'axis_1' / 'axis_2'         : str, primary and secondary axes used
        'axis_3'                    : str, thickness (ignored) axis
        'variances'                 : dict, variance per axis
        'gaps_axis1' / 'gaps_axis2' : list, all detected gap pairs
        'inlier_mask'               : ndarray, SOR mask (True = kept)
        'num_filled'                : int, total generated fill points
        'avg_point_spacing'         : float, auto-tuned spacing estimate
        'slice_thickness'           : float, effective slice thickness used
        'gap_threshold'             : float, effective gap threshold used
    """
    pts = np.asarray(points, dtype=np.float64)

    # ---- Step 0: SOR ----
    if verbose:
        print(f"[SOR] Input: {len(pts)} points")
    inlier_mask = statistical_outlier_removal(pts, k=sor_k, std_multiplier=sor_std_multiplier)
    pts_clean   = pts[inlier_mask]
    if verbose:
        print(f"[SOR] After outlier removal: {len(pts_clean)} points  "
              f"({len(pts_clean)/len(pts)*100:.1f}%)")

    # ---- Auto-Tuning: KDTree-based avg_point_spacing ----
    # Build cKDTree on inlier cloud; query k=2 (self + 1st NN), take 2nd distance
    if verbose:
        print("[AutoTune] Computing avg_point_spacing via cKDTree (k=2) ...")
    ckdtree = cKDTree(pts_clean)
    nn_distances, _ = ckdtree.query(pts_clean, k=2)   # col-0 is self (dist≈0), col-1 is 1st NN
    avg_point_spacing = float(np.mean(nn_distances[:, 1]))
    if verbose:
        print(f"[AutoTune] avg_point_spacing = {avg_point_spacing:.6f}")

    if slice_thickness is None:
        slice_thickness = avg_point_spacing * 2.0
        if verbose:
            print(f"[AutoTune] slice_thickness=None  →  set to {slice_thickness:.6f}  "
                  f"(= avg_point_spacing × 2)")
    else:
        if verbose:
            print(f"[AutoTune] slice_thickness={slice_thickness:.6f}  (user-provided)")

    if gap_threshold is None:
        gap_threshold = avg_point_spacing * 5.0
        if verbose:
            print(f"[AutoTune] gap_threshold=None  →  set to {gap_threshold:.6f}  "
                  f"(= avg_point_spacing × 5)")
    else:
        if verbose:
            print(f"[AutoTune] gap_threshold={gap_threshold:.6f}  (user-provided)")

    # ---- Step 1: Axis selection ----
    axis_1, axis_2, axis_3, variances = compute_variance_along_axes(pts_clean)
    if verbose:
        print(f"[Axis] Variance: X={variances['X']:.6f}  "
              f"Y={variances['Y']:.6f}  Z={variances['Z']:.6f}")
        print(f"[Axis] Primary={axis_1}  Secondary={axis_2}  "
              f"Thickness(ignored)={axis_3}")

    # ---- Step 2: Gap detection – Axis 1 ----
    if verbose:
        print(f"[Axis1] Processing {axis_1}-axis slicing ...")
    gaps_axis1 = process_axis(pts_clean, axis_1, slice_thickness, gap_threshold)
    if verbose:
        print(f"[Axis1] Detected {len(gaps_axis1)} gap pairs")

    # ---- Step 2: Gap detection – Axis 2 ----
    if verbose:
        print(f"[Axis2] Processing {axis_2}-axis slicing ...")
    gaps_axis2 = process_axis(pts_clean, axis_2, slice_thickness, gap_threshold)
    if verbose:
        print(f"[Axis2] Detected {len(gaps_axis2)} gap pairs")

    # ---- Step 3: Bezier fill ----
    if verbose:
        print(f"[Bezier] Filling gaps ...")
    bezier_pts_1 = fill_all_gaps(pts_clean, gaps_axis1,
                                 num_points_per_gap=num_points_per_gap,
                                 neighbor_k=neighbor_k)
    bezier_pts_2 = fill_all_gaps(pts_clean, gaps_axis2,
                                 num_points_per_gap=num_points_per_gap,
                                 neighbor_k=neighbor_k)
    bezier_all   = np.vstack([bezier_pts_1, bezier_pts_2]) if len(bezier_pts_1) or len(bezier_pts_2) else np.empty((0,3))

    if verbose:
        print(f"[Bezier] Generated {len(bezier_all)} fill points "
              f"({len(bezier_pts_1)} from {axis_1}-axis, "
              f"{len(bezier_pts_2)} from {axis_2}-axis)")

    # ---- Step 4: Merge / average ----
    merged = merge_and_average_points(pts_clean, bezier_all)

    if verbose:
        print(f"[Merge] Final point count: {len(merged)}")

    return {
        'original_inlier_points': pts_clean,
        'bezier_pts_axis1': bezier_pts_1,
        'bezier_pts_axis2': bezier_pts_2,
        'combined_bezier_pts': bezier_all,
        'merged_points': merged,
        'axis_1': axis_1,
        'axis_2': axis_2,
        'axis_3': axis_3,
        'variances': variances,
        'gaps_axis1': gaps_axis1,
        'gaps_axis2': gaps_axis2,
        'inlier_mask': inlier_mask,
        'num_filled': len(bezier_all),
        'avg_point_spacing': avg_point_spacing,
        'slice_thickness': slice_thickness,
        'gap_threshold': gap_threshold,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser(description='3D Point Cloud Hole Filling')
    parser.add_argument('input',  help='Input .xyz / .txt point cloud file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output .xyz file (default: <input>_filled.xyz)')
    parser.add_argument('--slice_thickness', type=float, default=None)
    parser.add_argument('--gap_threshold',    type=float, default=None)
    parser.add_argument('--num_points',       type=int,   default=20)
    parser.add_argument('--neighbor_k',       type=int,   default=5)
    parser.add_argument('--sor_k',            type=int,   default=10)
    parser.add_argument('--sor_std',          type=float, default=2.0)
    args = parser.parse_args()

    # Load points
    data = np.loadtxt(args.input)
    points = data[:, :3]

    result = process_point_cloud(
        points,
        slice_thickness=args.slice_thickness,
        gap_threshold=args.gap_threshold,
        num_points_per_gap=args.num_points,
        neighbor_k=args.neighbor_k,
        sor_k=args.sor_k,
        sor_std_multiplier=args.sor_std,
    )

    # Save merged output
    out_path = args.output or (os.path.splitext(args.input)[0] + '_filled.xyz')
    np.savetxt(out_path, result['merged_points'], fmt='%.8f %.8f %.8f')
    print(f'Saved → {out_path}  ({len(result["merged_points"])} points)')
