def inside_point(point, polygon):
    """
    Check if a point is inside a polygon.

    :param point: A tuple representing the (x, y) coordinates of the point.
    :param polygon: A list of tuples representing the vertices of the polygon in order.
    :return: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False  

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside
ans = inside_point((3, 3), [(0, 0), (5, 0), (5, 5), (0, 5)])  # Should return True
print(ans)