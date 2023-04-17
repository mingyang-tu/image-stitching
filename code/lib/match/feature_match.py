import numpy as np
import cv2
from BBF import KDTree
import time


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


if __name__ == "__main__":
    root = "../../../test/"
    names = [
        "100-0038_img.jpg",
        "100-0039_img.jpg",
        "100-0040_img.jpg"
    ]

    images = []
    for i in names:
        images.append(cv2.imread(root + i))

    img_match = np.concatenate(images, axis=1)

    SHIFT1 = images[0].shape[1]
    SHIFT2 = SHIFT1 + images[1].shape[1]

    sift = cv2.SIFT_create()

    kps = []
    descs = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        kps.append([i.pt for i in kp])
        descs.append(des)

    start = time.time()
    pair = feature_match(kps, descs)
    end = time.time()

    print(f"Ellapsed time: {end-start:.4f} s")

    for (x1, y1), (x2, y2) in pair[0]:
        cv2.line(
            img_match,
            (round(x1), round(y1)),
            (round(x2)+SHIFT1, round(y2)),
            (255, 0, 0), 1
        )
        cv2.circle(
            img_match,
            center=(round(x1), round(y1)),
            radius=3, color=(0, 0, 255), thickness=-1
        )
        cv2.circle(
            img_match,
            center=(round(x2)+SHIFT1, round(y2)),
            radius=3, color=(0, 0, 255), thickness=-1
        )
    for (x1, y1), (x2, y2) in pair[1]:
        cv2.line(
            img_match,
            (round(x1)+SHIFT1, round(y1)),
            (round(x2)+SHIFT2, round(y2)),
            (255, 0, 0), 1
        )
        cv2.circle(
            img_match,
            center=(round(x1)+SHIFT1, round(y1)),
            radius=3, color=(0, 0, 255), thickness=-1
        )
        cv2.circle(
            img_match,
            center=(round(x2)+SHIFT2, round(y2)),
            radius=3, color=(0, 0, 255), thickness=-1
        )  

    cv2.imshow("match", img_match)

    cv2.waitKey()
    cv2.destroyAllWindows()
