"""
3D Point Cloud Hole Filling via Dual-Axis Slicing & Bezier Cross-Hatching

Algorithm Pipeline:
  1. Statistical Outlier Removal (SOR)
  2. Axis Selection via Variance Detecter
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
# 2. PCA ALIGNMENT – ROTATION TO PRINCIPAL AXES
# =============================================================================

def pca_align(points):
    """
    Perform PCA to find principal axes and rotate the point cloud
    so that the axes of highest variance align with X and Y, 
    and the axis of least variance (surface normal) aligns with Z.

    Returns:
        pts_rotated: points in the new PCA-aligned coordinate system
        rotation_matrix: the 3x3 matrix used for rotation
        mean_pt: the centroid subtracted before rotation
    """
    pts = np.asarray(points)
    mean_pt = np.mean(pts, axis=0)
    pts_centered = pts - mean_pt

    # Compute Covariance Matrix
    cov = np.cov(pts_centered, rowvar=False)

    # Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalues descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    # Rotate points
    pts_rotated = pts_centered @ eigenvectors

    return pts_rotated, eigenvectors, mean_pt


def apply_inverse_pca(points_rotated, rotation_matrix, mean_pt):
    """Map points back from PCA space to original 3D space."""
    return (points_rotated @ rotation_matrix.T) + mean_pt


def compute_variance_along_axes(points):
    """Return variance for each axis sorted descending (legacy wrapper)."""
    pts = np.asarray(points)
    variances = {
        'X': float(np.var(pts[:, 0])),
        'Y': float(np.var(pts[:, 1])),
        'Z': float(np.var(pts[:, 2])),
    }
    sorted_axes = sorted(variances, key=variances.get, reverse=True)
    return sorted_axes[0], sorted_axes[1], sorted_axes[2], variances


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


def compute_slope_vector(point, neighbors, opposite_point=None):
    """
    Compute the surface slope direction at a gap boundary point.
    
    Algorithm:
    1. Filter neighbors to only those "behind" the boundary (surface side).
    2. Fit a line: direction from centroid of behind-neighbors through the
       boundary point, pointing into the gap.
    
    Parameters
    ----------
    point          : (3,) boundary point at the edge of the gap
    neighbors      : (N, 3) nearby points
    opposite_point : (3,) boundary point on the OTHER side of the gap
    
    Returns
    -------
    (3,) unit vector pointing from surface into the gap
    """
    if len(neighbors) == 0:
        return np.zeros(3)

    # Filter to "behind" neighbors (surface side, away from gap)
    filtered = neighbors
    if opposite_point is not None:
        gap_dir = opposite_point - point
        gap_norm = np.linalg.norm(gap_dir)
        if gap_norm > 1e-12:
            gap_unit = gap_dir / gap_norm
            dots = (neighbors - point) @ gap_unit
            behind = neighbors[dots < 0]
            if len(behind) >= 2:
                filtered = behind

    # Direction: centroid of behind-neighbors → boundary point → into gap
    centroid = np.mean(filtered, axis=0)
    direction = point - centroid
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-12:
        # Fallback: just point toward opposite
        if opposite_point is not None:
            direction = opposite_point - point
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                return np.zeros(3)
        else:
            return np.zeros(3)
    
    return direction / dir_norm


# Backward-compatible wrapper (used in app.py visualization)
def compute_tangent_from_neighbors(point, neighbors, gap_length=None, strength=0.33, opposite_point=None):
    """Returns a scaled tangent vector for visualization."""
    unit_dir = compute_slope_vector(point, neighbors, opposite_point=opposite_point)
    if gap_length is not None and gap_length > 0:
        return unit_dir * (strength * gap_length)
    return unit_dir * strength


def find_apex_point(p_left, v_left, p_right, v_right):
    """
    Triangulation: find the apex point p_c where two rays intersect.
    
    Ray 1: p_left  + t * v_left
    Ray 2: p_right + s * v_right
    
    In 3D, rays rarely intersect exactly. We find the midpoint of the
    shortest segment connecting the two rays (closest approach).
    
    The apex is clamped to at most 50% of the gap distance from the
    midpoint to prevent extreme curvature.
    
    Parameters
    ----------
    p_left  : (3,) left boundary point
    v_left  : (3,) slope direction at left (unit vector)
    p_right : (3,) right boundary point
    v_right : (3,) slope direction at right (unit vector)
    
    Returns
    -------
    p_c : (3,) apex point
    """
    w0 = p_left - p_right
    a = float(np.dot(v_left, v_left))
    b = float(np.dot(v_left, v_right))
    c = float(np.dot(v_right, v_right))
    d = float(np.dot(v_left, w0))
    e = float(np.dot(v_right, w0))
    
    denom = a * c - b * b
    
    gap_vec = p_right - p_left
    gap_length = float(np.linalg.norm(gap_vec))
    midpoint = (p_left + p_right) / 2.0
    
    if abs(denom) < 1e-12:
        # Rays are parallel — fallback: midpoint raised perpendicular
        perp = np.cross(gap_vec, v_left)
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-12:
            perp = perp / perp_norm
            return midpoint + perp * (gap_length * 0.3)
        else:
            return midpoint
    
    t_param = (b * e - c * d) / denom
    s_param = (a * e - b * d) / denom
    
    # Ensure rays go "forward" (positive parameters only)
    t_param = max(t_param, 0.0)
    s_param = max(s_param, 0.0)
    
    closest_on_ray1 = p_left + t_param * v_left
    closest_on_ray2 = p_right + s_param * v_right
    
    # Apex = midpoint between the two closest points
    p_c = (closest_on_ray1 + closest_on_ray2) / 2.0
    
    # CLAMP: limit apex distance to max 50% of gap distance
    apex_offset = np.linalg.norm(p_c - midpoint)
    max_offset = gap_length * 0.5
    if apex_offset > max_offset and apex_offset > 1e-12:
        p_c = midpoint + (p_c - midpoint) * (max_offset / apex_offset)
    
    return p_c


def cubic_bezier_fill_gap(p_left, p_right, num_points=20, neighbors_left=None, neighbors_right=None):
    """
    Fill a gap between p_left and p_right with a cubic Bezier curve.
    
    Algorithm (from research poster):
    
    Step 1 — Triangulation:
        Compute slope vectors v_l, v_r from neighbours at each boundary.
        Find apex point p_c where the two rays intersect (triangulation).
    
    Step 2 — Control Point Calculation (1:2 ratio):
        P1 = (2/3) * p_c + (1/3) * p_left
        P2 = (1/3) * p_c + (2/3) * p_right
    
    Bezier curve:  B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
    
    Parameters
    ----------
    p_left, p_right : array-like (3,)
    num_points      : int, number of interpolated points
    neighbors_left  : array (N, 3)
    neighbors_right : array (N, 3)
    
    Returns
    -------
    curve_pts : ndarray (num_points, 3)
    """
    p_left  = np.asarray(p_left,  dtype=np.float64)
    p_right = np.asarray(p_right, dtype=np.float64)

    # Step 1: Compute slope vectors (direction from surface into gap)
    v_left = compute_slope_vector(
        p_left,
        neighbors_left if neighbors_left is not None else np.empty((0, 3)),
        opposite_point=p_right
    )
    v_right = compute_slope_vector(
        p_right,
        neighbors_right if neighbors_right is not None else np.empty((0, 3)),
        opposite_point=p_left
    )

    # Step 1: Triangulation — find apex point p_c
    p_c = find_apex_point(p_left, v_left, p_right, v_right)

    # Step 2: Control points with 1:2 ratio
    P0 = p_left
    P1 = (2.0 / 3.0) * p_c + (1.0 / 3.0) * p_left    # closer to apex
    P2 = (1.0 / 3.0) * p_c + (2.0 / 3.0) * p_right   # closer to boundary
    P3 = p_right

    # Bezier curve: B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
    t_vals = np.linspace(0.0, 1.0, num_points)
    curve_pts = np.zeros((num_points, 3), dtype=np.float64)

    for i, t in enumerate(t_vals):
        u = 1.0 - t
        curve_pts[i] = u**3*P0 + 3*u**2*t*P1 + 3*u*t**2*P2 + t**3*P3

    return curve_pts


# =============================================================================
# 5. CROSS-HATCHING SURFACE GENERATION
# =============================================================================

def merge_and_average_points(original, new_points, merge_distance=1e-3):
    """
    Combine original + generated points, averaging near-duplicates.
    Uses KDTree to merge points within merge_distance.
    """
    orig_pts = np.asarray(original)
    if len(new_points) == 0:
        return orig_pts

    new_pts = np.asarray(new_points)
    all_pts = np.vstack([orig_pts, new_pts])
    tree    = KDTree(all_pts)

    merged = []
    used   = np.zeros(len(all_pts), dtype=bool)
    
    for i in range(len(all_pts)):
        if used[i]:
            continue
        
        # Find all points within merge_distance
        indices = tree.query_ball_point(all_pts[i], merge_distance)
        
        # Mark them as used
        used[indices] = True
        
        # Average the cluster
        merged.append(np.mean(all_pts[indices], axis=0))

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
                        use_sor=True,
                        use_pca=True,
                        use_cross_hatch=True,
                        fill_method='bezier',
                        verbose=True):
    """
    Complete hole-filling pipeline with PCA alignment for robust detection.

    Ablation parameters
    -------------------
    use_sor        : bool — enable Statistical Outlier Removal
    use_pca        : bool — enable PCA alignment
    use_cross_hatch: bool — if False, only use primary axis (no dual-axis)
    fill_method    : str  — 'bezier', 'linear', 'bspline', 'nearest'
    """
    import time as _time
    timings = {}

    pts = np.asarray(points, dtype=np.float64)

    # ---- Step 0: PCA Alignment ----
    t0 = _time.perf_counter()
    if use_pca:
        if verbose:
            print("[PCA] Aligning point cloud to principal axes...")
        pts_pca, rotation_matrix, mean_pt = pca_align(pts)
    else:
        if verbose:
            print("[PCA] SKIPPED (ablation)")
        pts_pca = pts.copy()
        rotation_matrix = np.eye(3)
        mean_pt = np.zeros(3)
    timings['pca'] = _time.perf_counter() - t0

    # ---- Step 1: SOR in PCA Space ----
    t0 = _time.perf_counter()
    if use_sor:
        if verbose:
            print(f"[SOR] Input: {len(pts_pca)} points")
        inlier_mask = statistical_outlier_removal(pts_pca, k=sor_k, std_multiplier=sor_std_multiplier)
        pts_clean   = pts_pca[inlier_mask]
        if verbose:
            print(f"[SOR] After outlier removal: {len(pts_clean)} points  "
                  f"({len(pts_clean)/len(pts_pca)*100:.1f}%)")
    else:
        if verbose:
            print("[SOR] SKIPPED (ablation)")
        inlier_mask = np.ones(len(pts_pca), dtype=bool)
        pts_clean = pts_pca
    timings['sor'] = _time.perf_counter() - t0

    # ---- Auto-Tuning: KDTree-based avg_point_spacing ----
    t0 = _time.perf_counter()
    if verbose:
        print("[AutoTune] Computing avg_point_spacing via cKDTree (k=2) ...")
    ckdtree = cKDTree(pts_clean)
    nn_distances, _ = ckdtree.query(pts_clean, k=2)   # col-0 is self (dist≈0), col-1 is 1st NN
    avg_point_spacing = float(np.mean(nn_distances[:, 1]))
    if verbose:
        print(f"[AutoTune] avg_point_spacing = {avg_point_spacing:.6f}")

    if slice_thickness is None:
        slice_thickness = avg_point_spacing * 2.0
    if gap_threshold is None:
        gap_threshold = avg_point_spacing * 5.0
    timings['auto_tune'] = _time.perf_counter() - t0

    # In PCA space, the primary/secondary/thickness axes are 0, 1, 2 (X, Y, Z)
    axis_1, axis_2, axis_3 = 'X', 'Y', 'Z'

    # ---- Step 2: Gap detection ----
    t0 = _time.perf_counter()
    if verbose:
        print(f"[Axis1] Processing {axis_1}-axis slicing (PCA Space) ...")
    gaps_axis1 = process_axis(pts_clean, axis_1, slice_thickness, gap_threshold)

    if use_cross_hatch:
        if verbose:
            print(f"[Axis2] Processing {axis_2}-axis slicing (PCA Space) ...")
        gaps_axis2 = process_axis(pts_clean, axis_2, slice_thickness, gap_threshold)
    else:
        if verbose:
            print("[Axis2] SKIPPED (single-axis ablation)")
        gaps_axis2 = []
    timings['gap_detection'] = _time.perf_counter() - t0

    # ---- Step 3: Fill gaps ----
    t0 = _time.perf_counter()

    # Select fill function based on fill_method
    if fill_method == 'bezier':
        fill_fn = fill_all_gaps
        if verbose:
            print(f"[Fill] Filling gaps with Cubic Bezier ...")
    else:
        from comparative import fill_all_gaps_with_method
        if verbose:
            print(f"[Fill] Filling gaps with method: {fill_method} ...")
        fill_fn = lambda pts, gaps, **kw: fill_all_gaps_with_method(
            pts, gaps, method=fill_method, **kw)

    bezier_pts_1_pca = fill_fn(pts_clean, gaps_axis1,
                               num_points_per_gap=num_points_per_gap,
                               neighbor_k=neighbor_k)
    if use_cross_hatch and gaps_axis2:
        bezier_pts_2_pca = fill_fn(pts_clean, gaps_axis2,
                                   num_points_per_gap=num_points_per_gap,
                                   neighbor_k=neighbor_k)
    else:
        bezier_pts_2_pca = np.empty((0, 3))
    timings['fill'] = _time.perf_counter() - t0

    # ---- Step 4: Inverse Transform back to Original Space ----
    t0 = _time.perf_counter()
    if verbose:
        print("[PCA] Mapping results back to original coordinate system...")

    pts_clean_orig = apply_inverse_pca(pts_clean, rotation_matrix, mean_pt)

    bezier_pts_1_orig = apply_inverse_pca(bezier_pts_1_pca, rotation_matrix, mean_pt) if len(bezier_pts_1_pca) > 0 else np.empty((0,3))
    bezier_pts_2_orig = apply_inverse_pca(bezier_pts_2_pca, rotation_matrix, mean_pt) if len(bezier_pts_2_pca) > 0 else np.empty((0,3))

    bezier_all_orig = np.vstack([bezier_pts_1_orig, bezier_pts_2_orig]) if len(bezier_pts_1_orig) or len(bezier_pts_2_orig) else np.empty((0,3))

    # Convert gap boundary points back to original space
    def get_boundary_pts_orig(gaps):
        if not gaps:
            return np.empty((0, 3))
        b_pts_pca = []
        for p_left, p_right, _, _ in gaps:
            b_pts_pca.append(p_left)
            b_pts_pca.append(p_right)
        b_pts_pca = np.array(b_pts_pca)
        return apply_inverse_pca(b_pts_pca, rotation_matrix, mean_pt)

    boundary_pts_axis1_orig = get_boundary_pts_orig(gaps_axis1)
    boundary_pts_axis2_orig = get_boundary_pts_orig(gaps_axis2)
    boundary_pts_all_orig = np.vstack([boundary_pts_axis1_orig, boundary_pts_axis2_orig]) if len(boundary_pts_axis1_orig) or len(boundary_pts_axis2_orig) else np.empty((0,3))

    # Merge in original space with a dynamic merge distance based on average spacing
    merge_dist = avg_point_spacing * 0.2
    merged_orig = merge_and_average_points(pts_clean_orig, bezier_all_orig, merge_distance=merge_dist)
    timings['merge'] = _time.perf_counter() - t0
    timings['total'] = sum(timings.values())

    if verbose:
        print(f"[Merge] merge_distance={merge_dist:.6f}")
        print(f"[Merge] Final point count: {len(merged_orig)}")
        print(f"[Timing] {timings}")

    return {
        'original_inlier_points': pts_clean_orig,
        'bezier_pts_axis1': bezier_pts_1_orig,
        'bezier_pts_axis2': bezier_pts_2_orig,
        'combined_bezier_pts': bezier_all_orig,
        'boundary_pts_axis1': boundary_pts_axis1_orig,
        'boundary_pts_axis2': boundary_pts_axis2_orig,
        'combined_boundary_pts': boundary_pts_all_orig,
        'merged_points': merged_orig,
        'axis_1': axis_1,
        'axis_2': axis_2,
        'axis_3': axis_3,
        'variances': {'PCA_X': 0, 'PCA_Y': 0, 'PCA_Z': 0},
        'gaps_axis1': gaps_axis1,
        'gaps_axis2': gaps_axis2,
        'inlier_mask': inlier_mask,
        'num_filled': len(bezier_all_orig),
        'avg_point_spacing': avg_point_spacing,
        'slice_thickness': slice_thickness,
        'gap_threshold': gap_threshold,
        'timings': timings,
        'fill_method': fill_method,
        'pts_clean_pca': pts_clean,
        'rotation_matrix': rotation_matrix,
        'mean_pt': mean_pt,
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
    parser.add_argument('--fill_method',      type=str,   default='bezier',
                        choices=['bezier', 'linear', 'bspline', 'nearest'])
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
        fill_method=args.fill_method,
    )

    # Save merged output
    out_path = args.output or (os.path.splitext(args.input)[0] + '_filled.xyz')
    np.savetxt(out_path, result['merged_points'], fmt='%.8f %.8f %.8f')
    print(f'Saved → {out_path}  ({len(result["merged_points"])} points)')

