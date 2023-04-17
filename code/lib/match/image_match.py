import numpy as np


def ransac(pair, k=50, threshold=5):
    max_inlier = 0
    best_shift = (0, 0)
    for _ in range(k):
        select = np.random.randint(0, len(pair))
        (x1, y1), (x2, y2) = pair[select]
        dx, dy = x1 - x2, y1 - y2

        inlier = 0
        dx_sum, dy_sum = 0, 0
        for (x1, y1), (x2, y2) in pair:
            _dx, _dy = x1 - x2, y1 - y2
            if ((_dx - dx) ** 2 + (_dy - dy) ** 2) ** (1/2) <= threshold:
                inlier += 1
                dx_sum += _dx
                dy_sum += _dy
        if inlier > max_inlier:
            max_inlier = inlier
            best_shift = (dx_sum / inlier, dy_sum / inlier)
    print(f"Ratio of inliers: {max_inlier}/{len(pair)} = {max_inlier / len(pair) * 100:.2f}%")
    return best_shift
