from .BBF import KDTree
import numpy as np
import time


def feature_match(keypoints, descriptions, threshold=10):
    N = len(keypoints)
    lengths = [[0 for j in range(N)] for i in range(N)]
    offsets = [[(float("inf"), float("inf")) for j in range(N)] for i in range(N)]

    print("-- Start Matching --")
    start = time.time()
    for i in range(N):
        kp1, des1 = keypoints[i], descriptions[i]
        tree1 = KDTree()
        tree1.build_tree(des1)

        for j in range(i+1, N):
            print(f"Matching image {i} -> {j}", end="\r")
            kp2, des2 = keypoints[j], descriptions[j]
            pair = _feature_match(kp1, kp2, tree1, des2)
            if len(pair) >= threshold:
                lengths[i][j] = lengths[j][i] = len(pair)
                offsets[i][j] = ransac(pair)
                offsets[j][i] = (-offsets[i][j][0], -offsets[i][j][1])
    end = time.time()
    print(f"Cost {end-start:.2f} (s)")

    return lengths, offsets


def _feature_match(kp1, kp2, tree1, des2, ratio=0.6):
    pair = []
    for i in range(len(kp2)):
        m2, m1 = tree1.knn(des2[i, :], 2, 100)
        if m1.dist < m2.dist * ratio:
            pair.append((kp1[m1.index], kp2[i]))
    return pair


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
            best_shift = (round(dx_sum / inlier), round(dy_sum / inlier))
    print(f"Ratio of inliers: {max_inlier}/{len(pair)} = {max_inlier / len(pair) * 100:.2f}%")
    return best_shift
