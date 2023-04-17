import cv2
from .BBF import KDTree


def _feature_match(kp1, des1, kp2, des2, ratio=0.5):
    pair = []
    tree = KDTree()
    tree.build_tree(des1)
    for i in range(len(kp2)):
        m2, m1 = tree.knn(des2[i, :], 2, 100)
        if m1.dist < m2.dist * ratio:
            pair.append((kp1[m1.index], kp2[i]))
    return pair


def feature_match(keypoints, descriptions):
    N = len(keypoints)
    matched = []
    for i in range(N-1):
        kp1, des1 = keypoints[i], descriptions[i]
        kp2, des2 = keypoints[i+1], descriptions[i+1]
        matched.append(_feature_match(kp1, des1, kp2, des2))
    return matched


def display_matching(concat_image, pairs, shifts):
    for i in range(len(pairs)):
        print(f"Valid pairs: {len(pairs[i])}")
        for (x1, y1), (x2, y2) in pairs[i]:
            cv2.line(
                concat_image,
                (round(x1)+shifts[i], round(y1)),
                (round(x2)+shifts[i+1], round(y2)),
                (255, 0, 0), 1
            )
            cv2.circle(
                concat_image,
                center=(round(x1)+shifts[i], round(y1)),
                radius=3, color=(0, 0, 255), thickness=-1
            )
            cv2.circle(
                concat_image,
                center=(round(x2)+shifts[i+1], round(y2)),
                radius=3, color=(0, 0, 255), thickness=-1
            )
    cv2.imshow("match", concat_image)

    cv2.waitKey()
    cv2.destroyAllWindows()
