"""
3D Point Cloud Hole Filling via Dual-Axis Slicing & Bezier Cross-Hatching

Algorithm Pipeline:
  1. Axis Selection via Variance Detecter
  2. Dual-Axis Slicing & 1D Gap Detection
  3. Slice-by-Slice Cubic Bezier Curve Filling
  4. Cross-Hatching Surface Generation (merge both axes)
"""

import numpy as np
from scipy.spatial import cKDTree, KDTree



# =============================================================================
# 1. PCA ALIGNMENT – ROTATION TO PRINCIPAL AXES
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


AXIS_TO_COL = {'X': 0, 'Y': 1, 'Z': 2}


def axis_col(axis_name):
    return AXIS_TO_COL[axis_name]


# =============================================================================
# 2. DUAL-AXIS SLICING & 1D GAP DETECTION
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


def filter_gaps_by_boundary_neighbors(gaps, radius, min_neighbors=3):
    """
    Filter gaps by checking if their boundary points (p_left, p_right) have enough
    neighboring boundary points. A true hole boundary should be contiguous across slices.
    """
    if not gaps:
        return []

    # 1. Collect all boundary points into a KDTree
    b_pts = []
    for g in gaps:
        b_pts.append(g[0]) # p_left
        b_pts.append(g[1]) # p_right
    b_pts = np.array(b_pts)
    
    # We already imported cKDTree at the top. Use it.
    from scipy.spatial import cKDTree
    tree = cKDTree(b_pts)

    valid_gaps = []
    for g in gaps:
        p_l = g[0]
        p_r = g[1]
        
        # 2. Count neighbors. KDTree.query_ball_point returns list of neighbor indices.
        # Includes the point itself, so count >= 1 always.
        count_l = len(tree.query_ball_point(p_l, r=radius))
        count_r = len(tree.query_ball_point(p_r, r=radius))
        
        # 3. If BOTH sides have neighbors, it's a valid gap (part of a real hole)
        if count_l >= min_neighbors and count_r >= min_neighbors:
            valid_gaps.append(g)

    return valid_gaps





def estimate_tangent_2d(pts_2d, point_2d, opposite_2d=None):
    """
    Tangent estimation via average of vectors from the boundary point
    to its 3 closest neighbors.

    Steps:
      1. Pick the 3 nearest neighbors from the surface side (behind the gap)
      2. Compute vectors: neighbor - point, for each of the 3
      3. Average those 3 vectors → tangent direction
      4. Orient the tangent to point INTO the gap
    """
    pts = np.asarray(pts_2d, dtype=np.float64)
    point = np.asarray(point_2d, dtype=np.float64)

    if len(pts) < 1:
        return np.zeros(2)

    # Filter to "behind" points (surface side, opposite to gap)
    behind = pts
    if opposite_2d is not None:
        opposite = np.asarray(opposite_2d, dtype=np.float64)
        gap_dir = opposite - point
        gap_norm = np.linalg.norm(gap_dir)
        if gap_norm > 1e-12:
            gap_unit = gap_dir / gap_norm
            dots = (pts - point) @ gap_unit
            behind_mask = dots < 0
            if np.sum(behind_mask) >= 1:
                behind = pts[behind_mask]

    # Take the 3 closest neighbors (or fewer if not enough)
    dists = np.linalg.norm(behind - point, axis=1)
    order = np.argsort(dists)
    k = min(3, len(behind))
    nearest = behind[order[:k]]

    # Average of vectors from the point to each neighbor
    vectors = nearest - point
    tangent = np.mean(vectors, axis=0)

    # Normalize
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return np.zeros(2)
    tangent = tangent / norm

    # Orient the tangent to point INTO the gap
    if opposite_2d is not None:
        gap_dir = np.asarray(opposite_2d, dtype=np.float64) - point
        if np.dot(tangent, gap_dir) < 0:
            tangent = -tangent

    return tangent

def find_apex_2d(p_l, v_l, p_r, v_r, gap_length):
    """
    2D ray intersection for the apex.
    Returns apex point if rays intersect properly, None otherwise.
    """
    A = np.column_stack([v_l, -v_r])
    b = p_r - p_l
    
    try:
        ts = np.linalg.solve(A, b)
        t, s = ts[0], ts[1]
        if t > 0 and s > 0:
            apex = p_l + t * v_l
            # Check if apex is within reasonable range
            midpoint = (p_l + p_r) / 2.0
            offset = np.linalg.norm(apex - midpoint)
            if offset <= gap_length * 0.5:
                return apex
    except np.linalg.LinAlgError:
        pass
    
    return None

def compute_g1_angle(v_left, v_right):
    """Compute G1 continuity angle in degrees."""
    n_l = np.linalg.norm(v_left)
    n_r = np.linalg.norm(v_right)
    if n_l < 1e-12 or n_r < 1e-12:
        return 180.0
    cos_angle = np.dot(v_left / n_l, v_right / n_r)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def cubic_bezier_fill_gap_g1(p_left, p_right, slice_points, num_points=20):
    """
    Fills a gap using cubic Bezier with G1 continuity.

    Two modes:
    - Apex mode: rays intersect properly → P1 = tension×apex + (1-tension)×P0
    - Hermite mode: rays don't intersect → P1 = P0 + tension×(gap/3)×tangent
    Both guarantee G1 continuity.
    """
    p_l = np.asarray(p_left)
    p_r = np.asarray(p_right)
    gap_vec = p_r - p_l
    gap_length = np.linalg.norm(gap_vec)
    
    if gap_length < 1e-12:
        return np.array([p_l]), np.zeros(2), np.zeros(2), None, []

    # Filter neighbors near left and right (3 neighbors + 1 for self-match filtering)
    tree = KDTree(slice_points)
    _, idx_l = tree.query(p_l, k=min(4, len(slice_points)))
    _, idx_r = tree.query(p_r, k=min(4, len(slice_points)))

    nb_l = slice_points[[i for i in idx_l if not np.allclose(slice_points[i], p_l)]]
    nb_r = slice_points[[i for i in idx_r if not np.allclose(slice_points[i], p_r)]]

    v_l = estimate_tangent_2d(nb_l, p_l, opposite_2d=p_r)
    v_r = estimate_tangent_2d(nb_r, p_r, opposite_2d=p_l)
    
    apex = find_apex_2d(p_l, v_l, p_r, v_r, gap_length)
    
    P0 = p_l
    P3 = p_r
    
    if apex is not None:
        # === APEX MODE: rays intersect properly ===
        # apex lies on ray P0 + t*v_l, so P1-P0 ∝ v_l → G1 correct
        midpoint = (p_l + p_r) / 2.0
        offset = np.linalg.norm(apex - midpoint)
        bulge_ratio = offset / gap_length
        
        curve_tension = 0.666
        if bulge_ratio > 0.05:
            factor = np.clip((bulge_ratio - 0.05) / 0.20, 0.0, 1.0)
            curve_tension = 0.666 - (factor * 0.45)
        
        P1 = curve_tension * apex + (1.0 - curve_tension) * P0
        P2 = curve_tension * apex + (1.0 - curve_tension) * P3
    else:
        # === HERMITE MODE: gap/3 along tangent ===
        # P1 = P0 + (gap/3)×v_l → G1 correct by construction
        hermite_scale = gap_length / 3.0
        curve_tension = 0.666
        
        P1 = P0 + curve_tension * hermite_scale * v_l
        P2 = P3 + curve_tension * hermite_scale * v_r
    
    t_vals = np.linspace(0.0, 1.0, num_points)[:, np.newaxis]
    u = 1.0 - t_vals
    curve = u**3*P0 + 3*u**2*t_vals*P1 + 3*u*t_vals**2*P2 + t_vals**3*P3
    
    return curve, v_l, v_r, apex, [P1, P2]

def cubic_bezier_fill_gap(p_left_3d, p_right_3d, num_points=20, neighbors_left=None, neighbors_right=None, slice_axis_idx=0):
    """
    Wrapper that maps the 3D points into a 2D plane, calls the 2D native Bezier fill,
    and maps the generated curve back to 3D safely.
    """
    p_l = np.asarray(p_left_3d)
    p_r = np.asarray(p_right_3d)
    
    other_cols = [c for c in (0, 1, 2) if c != slice_axis_idx]
    
    p_left_2d = p_l[other_cols]
    p_right_2d = p_r[other_cols]
    
    all_neighbors = []
    if neighbors_left is not None and len(neighbors_left) > 0:
        all_neighbors.append(neighbors_left[:, other_cols])
    if neighbors_right is not None and len(neighbors_right) > 0:
        all_neighbors.append(neighbors_right[:, other_cols])
        
    if all_neighbors:
        slice_points_2d = np.vstack(all_neighbors)
    else:
        slice_points_2d = np.empty((0, 2))
        
    curve_2d, _, _, _, _ = cubic_bezier_fill_gap_g1(
        p_left_2d, p_right_2d, slice_points_2d, 
        num_points=num_points
    )
    
    # Map back to 3D
    curve_3d = np.zeros((len(curve_2d), 3))
    curve_3d[:, slice_axis_idx] = (p_l[slice_axis_idx] + p_r[slice_axis_idx]) / 2.0
    for i, col in enumerate(other_cols):
        curve_3d[:, col] = curve_2d[:, i]
        
    return curve_3d


# =============================================================================
# 4. CROSS-HATCHING SURFACE GENERATION
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




# =============================================================================
# 5. SURFACE DENSIFICATION — INTERMEDIATE CURVES & UNIFORM SCATTER
# =============================================================================

def fill_all_gaps_tracked(points, gap_pairs, num_points_per_gap=20,
                          neighbor_k=5, slice_axis_idx=0, avg_spacing=None):
    """
    Like fill_all_gaps but returns individual curves with metadata
    for grouping and surface densification.
    """
    pts_array = np.asarray(points)
    tree = KDTree(pts_array)
    curves = []

    for p_left, p_right, dist, axis_val in gap_pairs:
        _, idx_left  = tree.query(p_left,  k=min(neighbor_k + 1, len(pts_array)))
        _, idx_right = tree.query(p_right, k=min(neighbor_k + 1, len(pts_array)))
        nb_left  = pts_array[[i for i in idx_left  if not np.allclose(pts_array[i], p_left)]]
        nb_right = pts_array[[i for i in idx_right if not np.allclose(pts_array[i], p_right)]]

        n_pts = num_points_per_gap
        if avg_spacing is not None and avg_spacing > 0:
            n_pts = max(3, int(dist / avg_spacing) + 1)

        curve = cubic_bezier_fill_gap(
            p_left, p_right,
            num_points=n_pts,
            neighbors_left=nb_left,
            neighbors_right=nb_right,
            slice_axis_idx=slice_axis_idx
        )
        curves.append({
            'points': curve,
            'axis_val': axis_val,
            'p_left': p_left.copy(),
            'p_right': p_right.copy(),
        })

    return curves


def _resample_curve_3d(curve_pts, n_target):
    """Resample a 3D curve to exactly n_target points via arc-length interpolation."""
    pts = np.asarray(curve_pts)
    if len(pts) == n_target:
        return pts.copy()
    if len(pts) < 2:
        return np.tile(pts[0], (n_target, 1)) if len(pts) >= 1 else np.zeros((n_target, 3))

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total < 1e-15:
        return np.tile(pts[0], (n_target, 1))

    targets = np.linspace(0, total, n_target)
    resampled = np.empty((n_target, 3))
    for i, tl in enumerate(targets):
        idx = min(np.searchsorted(cum, tl, side='right') - 1, len(pts) - 2)
        idx = max(idx, 0)
        frac = (tl - cum[idx]) / seg_lengths[idx] if seg_lengths[idx] > 1e-15 else 0.0
        resampled[i] = (1 - frac) * pts[idx] + frac * pts[idx + 1]
    return resampled


def group_curves_by_hole(curves, slice_axis_idx, avg_spacing):
    """
    Group curves from adjacent slices that fill the same hole.
    Two curves are grouped if their gap midpoints (in non-slice dimensions)
    are within a proximity threshold.
    """
    if len(curves) < 2:
        return []

    other_cols = [c for c in (0, 1, 2) if c != slice_axis_idx]
    sorted_curves = sorted(curves, key=lambda c: c['axis_val'])
    max_dist = avg_spacing * 30

    groups = []
    assigned = set()

    for i in range(len(sorted_curves)):
        if i in assigned:
            continue
        group = [sorted_curves[i]]
        assigned.add(i)

        for j in range(i + 1, len(sorted_curves)):
            if j in assigned:
                continue
            last = group[-1]
            curr = sorted_curves[j]
            mid_last = (last['p_left'][other_cols] + last['p_right'][other_cols]) / 2.0
            mid_curr = (curr['p_left'][other_cols] + curr['p_right'][other_cols]) / 2.0
            if np.linalg.norm(mid_curr - mid_last) < max_dist:
                group.append(curr)
                assigned.add(j)

        if len(group) >= 2:
            groups.append(group)

    return groups


def densify_between_curves(curve_groups, slice_axis_idx, avg_spacing):
    """
    Create intermediate curves and scatter uniform random points
    between adjacent Bezier curves filling the same hole.

    For each pair of adjacent curves:
      1. Linearly interpolate intermediate curves (avg of endpoints)
      2. Between all rails, bilinear-scatter random points at ~avg_spacing density

    Returns
    -------
    extra_points : ndarray (M, 3)
    """
    extra = []

    for group in curve_groups:
        group.sort(key=lambda c: c['axis_val'])

        for k in range(len(group) - 1):
            pts1 = group[k]['points']
            pts2 = group[k + 1]['points']

            # Resample both curves to the same number of points
            n_target = max(len(pts1), len(pts2), 10)
            r1 = _resample_curve_3d(pts1, n_target)
            r2 = _resample_curve_3d(pts2, n_target)

            # Number of intermediate curves based on slice gap vs avg_spacing
            slice_gap = abs(group[k + 1]['axis_val'] - group[k]['axis_val'])
            n_interp = max(0, int(slice_gap / (avg_spacing * 2.0)) - 1)

            # Build all rails: original + interpolated
            rails = [r1]
            for m in range(1, n_interp + 1):
                t = m / (n_interp + 1)
                rails.append((1 - t) * r1 + t * r2)
            rails.append(r2)

            # Add intermediate curve points (skip originals at index 0 and -1)
            for m in range(1, len(rails) - 1):
                extra.append(rails[m])

            # Scatter uniform random points between adjacent rails
            for r_idx in range(len(rails) - 1):
                ra, rb = rails[r_idx], rails[r_idx + 1]
                for p in range(len(ra) - 1):
                    q00, q10, q01, q11 = ra[p], ra[p+1], rb[p], rb[p+1]
                    # Approximate quad area via cross product of diagonals
                    d1 = q11 - q00
                    d2 = q10 - q01
                    area = 0.5 * np.linalg.norm(np.cross(d1, d2))
                    n_pts = int(area / (avg_spacing ** 2))
                    if n_pts > 0:
                        uv = np.random.uniform(0, 1, (n_pts, 2))
                        u, v = uv[:, 0:1], uv[:, 1:2]
                        pts = ((1-u)*(1-v)*q00 + u*(1-v)*q10 +
                               (1-u)*v*q01 + u*v*q11)
                        extra.append(pts)

    if not extra:
        return np.empty((0, 3))
    return np.vstack(extra)




# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_point_cloud(points,
                        slice_thickness=None,
                        gap_threshold=None,
                        num_points_per_gap=20,
                        neighbor_k=5,
                        use_pca=True,
                        use_cross_hatch=True,
                        verbose=True):
    """
    Complete hole-filling pipeline with PCA alignment for robust detection.

    Ablation flags: use_pca, use_cross_hatch
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

    # ---- Step 1: Mapping ----
    pts_clean = pts_pca

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
    raw_gaps_axis1 = process_axis(pts_clean, axis_1, slice_thickness, gap_threshold)
    # Check if neighbors exist within 2x slice thickness (at least 1 other boundary point)
    filter_radius = slice_thickness * 2.0
    gaps_axis1 = filter_gaps_by_boundary_neighbors(raw_gaps_axis1, radius=filter_radius, min_neighbors=2)

    if use_cross_hatch:
        if verbose:
            print(f"[Axis2] Processing {axis_2}-axis slicing (PCA Space) ...")
        raw_gaps_axis2 = process_axis(pts_clean, axis_2, slice_thickness, gap_threshold)
        gaps_axis2 = filter_gaps_by_boundary_neighbors(raw_gaps_axis2, radius=filter_radius, min_neighbors=2)
    else:
        if verbose:
            print("[Axis2] SKIPPED (single-axis ablation)")
        gaps_axis2 = []
    timings['gap_detection'] = _time.perf_counter() - t0

    # ---- Step 3: Fill gaps ----
    t0 = _time.perf_counter()
    if verbose:
        print(f"[Fill] Filling gaps with Cubic Bezier + Surface Densification ...")

    # --- Axis 1: tracked fill + densify ---
    curves_1 = fill_all_gaps_tracked(pts_clean, gaps_axis1,
                                     num_points_per_gap=num_points_per_gap,
                                     neighbor_k=neighbor_k,
                                     slice_axis_idx=0,
                                     avg_spacing=avg_point_spacing)
    bezier_pts_1_pca = np.vstack([c['points'] for c in curves_1]) if curves_1 else np.empty((0, 3))

    groups_1 = group_curves_by_hole(curves_1, 0, avg_point_spacing)
    dense_1 = densify_between_curves(groups_1, 0, avg_point_spacing)
    if len(dense_1) > 0:
        if verbose:
            print(f"[Densify] Axis 1: +{len(dense_1)} surface points")
        bezier_pts_1_pca = np.vstack([bezier_pts_1_pca, dense_1])

    # --- Axis 2: tracked fill + densify ---
    if use_cross_hatch and gaps_axis2:
        curves_2 = fill_all_gaps_tracked(pts_clean, gaps_axis2,
                                         num_points_per_gap=num_points_per_gap,
                                         neighbor_k=neighbor_k,
                                         slice_axis_idx=1,
                                         avg_spacing=avg_point_spacing)
        bezier_pts_2_pca = np.vstack([c['points'] for c in curves_2]) if curves_2 else np.empty((0, 3))

        groups_2 = group_curves_by_hole(curves_2, 1, avg_point_spacing)
        dense_2 = densify_between_curves(groups_2, 1, avg_point_spacing)
        if len(dense_2) > 0:
            if verbose:
                print(f"[Densify] Axis 2: +{len(dense_2)} surface points")
            bezier_pts_2_pca = np.vstack([bezier_pts_2_pca, dense_2])
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
        'num_filled': len(bezier_all_orig),
        'avg_point_spacing': avg_point_spacing,
        'slice_thickness': slice_thickness,
        'gap_threshold': gap_threshold,
        'timings': timings,
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
    args = parser.parse_args()

    data = np.loadtxt(args.input)
    points = data[:, :3]

    result = process_point_cloud(
        points,
        slice_thickness=args.slice_thickness,
        gap_threshold=args.gap_threshold,
        num_points_per_gap=args.num_points,
        neighbor_k=args.neighbor_k,
    )

    # Save merged output
    out_path = args.output or (os.path.splitext(args.input)[0] + '_filled.xyz')
    np.savetxt(out_path, result['merged_points'], fmt='%.8f %.8f %.8f')
    print(f'Saved → {out_path}  ({len(result["merged_points"])} points)')
