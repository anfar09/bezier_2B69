import sys
input = sys.stdin.readline

def cuboid(p, axis, w):
    rectangle = []

    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')

    for pt in p:
        if axis == 1:
            x, y = pt[1], pt[2]
        elif axis == 2:
            x, y = pt[0], pt[2]
        else:
            x, y = pt[0], pt[1]

        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)

    if w <= 0:
        return rectangle

    x0 = xmin
    while x0 < xmax:
        x1 = min(x0 + w, xmax)
        rectangle.append([(x0, ymin), (x1, ymax)])
        x0 += w

    return rectangle

n = int(input())
point = []
for _ in range(n):
    x, y, z = map(float, input().split())
    point.append((x, y, z))

w = list(map(float, input().split()))

recx = cuboid(point, 1, w[0])
recy = cuboid(point, 2, w[1])
recz = cuboid(point, 3, w[2])

print("slice X axis")
for rec in recx:
    print("(", rec[0][0], ",", rec[0][1], ")", "(", rec[1][0], ",", rec[1][1], ")", sep="")

print("slice Y axis")
for rec in recy:
    print("(", rec[0][0], ",", rec[0][1], ")", "(", rec[1][0], ",", rec[1][1], ")", sep="")

print("slice Z axis")
for rec in recz:
    print("(", rec[0][0], ",", rec[0][1], ")", "(", rec[1][0], ",", rec[1][1], ")", sep="")