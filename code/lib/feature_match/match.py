from .BBF import KDTree


def feature_match(keypoints, descriptions):
    N = len(keypoints)
    pairs = [[[] for j in range(N)] for i in range(N)]

    for i in range(N):
        kp1, des1 = keypoints[i], descriptions[i]
        tree1 = KDTree()
        tree1.build_tree(des1)

        for j in range(i+1, N):
            print(f"Matching Image {i} -> {j}")
            kp2, des2 = keypoints[j], descriptions[j]
            pairs[i][j] = _feature_match(kp1, kp2, tree1, des2)

    return pairs


def _feature_match(kp1, kp2, tree1, des2, ratio=0.5):
    pair = []
    for i in range(len(kp2)):
        m2, m1 = tree1.knn(des2[i, :], 2, 100)
        if m1.dist < m2.dist * ratio:
            pair.append((kp1[m1.index], kp2[i]))
    return pair
