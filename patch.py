import numpy as np

def estimate_tangent_2d(pts_2d, point_2d, opposite_2d=None):
    """
    Robust tangent estimation using 2D PCA eigenvalue extraction.
    This avoids math asymptotes (infinity) on vertical surfaces.
    """
    pts = np.asarray(pts_2d)
    point = np.asarray(point_2d)
    
    if len(pts) < 2:
        return np.zeros(2)
        
    behind = pts
    if opposite_2d is not None:
        opposite = np.asarray(opposite_2d)
        gap_dir = opposite - point
        gap_norm = np.linalg.norm(gap_dir)
        if gap_norm > 1e-12:
            gap_unit = gap_dir / gap_norm
            dots = (pts - point) @ gap_unit
            behind_mask = dots < 0
            if np.sum(behind_mask) >= 2:
                behind = pts[behind_mask]
                
    cov = np.cov(behind.T)
    evals, evecs = np.linalg.eigh(cov)
    tangent = evecs[:, -1]  # eigenvector with largest eigenvalue
    
    if opposite_2d is not None:
        gap_dir = opposite_2d - point
        if np.dot(tangent, gap_dir) < 0:
            tangent = -tangent
            
    return tangent

def find_apex_2d(p_l, v_l, p_r, v_r, gap_length):
    """2D ray intersection for the apex."""
    # p_l + t * v_l = p_r + s * v_r
    # t * v_l - s * v_r = p_r - p_l
    A = np.column_stack([v_l, -v_r])
    b = p_r - p_l
    midpoint = (p_l + p_r) / 2.0
    
    try:
        ts = np.linalg.solve(A, b)
        t, s = ts[0], ts[1]
        if t > 0 and s > 0:
            apex = p_l + t * v_l
            # Check clamp
            offset = np.linalg.norm(apex - midpoint)
            max_offset = gap_length * 0.25
            if offset > max_offset:
                apex = midpoint + (apex - midpoint) * (max_offset / offset)
            return apex
    except np.linalg.LinAlgError:
        pass
        
    # Parallel or solving failed / backward
    perp = np.array([-v_l[1], v_l[0]])
    gap_vec = p_r - p_l
    if np.dot(perp, gap_vec) < 0:
        perp = -perp
    return midpoint + perp * (gap_length * 0.3)

def cubic_bezier_fill_gap_g1(p_left, p_right, slice_points, num_points=20, flat_angle_threshold=160.0, force_linear=False):
    """
    Fills a gap in 2D space.
    """
    p_l = np.asarray(p_left)
    p_r = np.asarray(p_right)
    gap_vec = p_r - p_l
    gap_length = np.linalg.norm(gap_vec)
    
    if force_linear:
        t_vals = np.linspace(0.0, 1.0, num_points)[:, np.newaxis]
        curve = p_l * (1 - t_vals) + p_r * t_vals
        return curve, np.zeros(2), np.zeros(2), (p_l+p_r)/2, []

    v_l = estimate_tangent_2d(slice_points, p_l, opposite_2d=p_r)
    v_r = estimate_tangent_2d(slice_points, p_r, opposite_2d=p_l)
    
    # Compute G1
    n_l, n_r = np.linalg.norm(v_l), np.linalg.norm(v_r)
    g1_angle = 180.0
    if n_l > 1e-12 and n_r > 1e-12:
        cos_ang = np.dot(v_l/n_l, v_r/n_r)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        g1_angle = np.degrees(np.arccos(cos_ang))
        
    if g1_angle >= flat_angle_threshold:
        t_vals = np.linspace(0.0, 1.0, num_points)[:, np.newaxis]
        curve = p_l * (1 - t_vals) + p_r * t_vals
        return curve, v_l, v_r, (p_l+p_r)/2, []
        
    apex = find_apex_2d(p_l, v_l, p_r, v_r, gap_length)
    
    P0 = p_l
    P1 = (2.0/3.0)*apex + (1.0/3.0)*p_l
    P2 = (1.0/3.0)*apex + (2.0/3.0)*p_r
    P3 = p_r
    
    t_vals = np.linspace(0.0, 1.0, num_points)[:, np.newaxis]
    u = 1.0 - t_vals
    curve = u**3*P0 + 3*u**2*t_vals*P1 + 3*u*t_vals**2*P2 + t_vals**3*P3
    
    return curve, v_l, v_r, apex, [P1, P2]
