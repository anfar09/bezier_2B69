def inside(point, rect):
    x = point[0]
    y = point[1]

    xmin = min(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
    xmax = max(rect[0][0], rect[1][0], rect[2][0], rect[3][0])

    ymin = min(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
    ymax = max(rect[0][1], rect[1][1], rect[2][1], rect[3][1])

    if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
    else:
        return False

rect = [(0,0), (5,0), (5,5), (0,5)]
print(inside((3,3), rect))  # True
print(inside((6,3), rect))  # False