import numpy as np

def load_points_from_file(filepath):
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = map(float, parts[:3])
                    points.append((x, y, z))
    return points

def get_bounds(points):
    pts = np.array(points)
    return {
        'x': (pts[:,0].min(), pts[:,0].max()),
        'y': (pts[:,1].min(), pts[:,1].max()),
        'z': (pts[:,2].min(), pts[:,2].max())
    }

def cuboid_slices(points, axis, w):
    pts = np.array(points)

    axis_map = {1:0, 2:1, 3:2}
    idx = axis_map[axis]

    vals = pts[:, idx]
    mn, mx = vals.min(), vals.max()

    slices = []
    cur = mn

    while cur < mx:
        nxt = min(cur + w, mx)
        mask = (vals >= cur) & (vals < nxt)
        subset = pts[mask]

        slices.append({
            "range": (cur, nxt),
            "points": subset.tolist(),
            "count": len(subset)
        })

        cur += w

    return slices

# ---------- HOLE DETECTION ----------
def max_gap_2d(points2d):
    if len(points2d) < 2:
        return 0.0

    points2d = sorted(points2d)
    mx = 0.0

    for i in range(len(points2d)-1):
        x1, y1 = points2d[i]
        x2, y2 = points2d[i+1]
        dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
        mx = max(mx, dist)

    return mx

def detect_holes_in_slices(slices, axis_char):
    gaps = []

    for s in slices:
        pts = np.array(s['points'])

        if len(pts) == 0:
            gaps.append(0)
            continue

        if axis_char == 'X':
            pts2d = pts[:, [1,2]]
        elif axis_char == 'Y':
            pts2d = pts[:, [0,2]]
        else:
            pts2d = pts[:, [0,1]]

        gap = max_gap_2d(pts2d.tolist())
        gaps.append(gap)

    mu = np.mean(gaps)

    results = []
    for i, s in enumerate(slices):
        results.append({
            "slice": i,
            "start": s['range'][0],
            "end": s['range'][1],
            "gap": gaps[i],
            "has_hole": gaps[i] > mu
        })

    return results, mu