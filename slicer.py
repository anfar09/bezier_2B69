def load_points_from_file(filepath):
    """อ่านไฟล์ point cloud (.xyz, .txt, .pts) คืนเป็น list of tuples"""
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