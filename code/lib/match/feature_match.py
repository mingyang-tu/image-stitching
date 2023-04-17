import numpy as np
import cv2
from BBF import KDTree
import time


def _feature_match(kp1, des1, kp2, des2, ratio=0.7):
    pair = []
    tree = KDTree()
    tree.build_tree(des1)
    for i in range(len(kp2)):
        m2, m1 = tree.knn(des2[i, :], 2, 100)
        if m1.dist < m2.dist * ratio:
            pair.append((kp1[m1.index], kp2[i]))
    return pair


if __name__ == "__main__":

    image1 = cv2.imread("../../../test/100-0024_img.jpg")
    image2 = cv2.imread("../../../test/100-0025_img.jpg")

    match = np.concatenate([image1, image2], axis=1)

    _, SHIFT, _ = image1.shape

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    pt1 = [i.pt for i in kp1]
    pt2 = [i.pt for i in kp2]

    start = time.time()
    pair = _feature_match(pt1, des1, pt2, des2)
    end = time.time()

    print(f"Ellapsed time: {end-start:.4f} s")
    print(f"Valid pairs: {len(pair)}")

    for (x1, y1), (x2, y2) in pair:
        cv2.line(
            match,
            (round(x1), round(y1)),
            (round(x2)+SHIFT, round(y2)),
            (255, 0, 0), 1
        )
    for (x1, y1), (x2, y2) in pair:
        cv2.circle(
            match,
            center=(round(x1), round(y1)),
            radius=3, color=(0, 0, 255), thickness=-1
        )
        cv2.circle(
            match,
            center=(round(x2)+SHIFT, round(y2)),
            radius=3, color=(0, 0, 255), thickness=-1
        )

    cv2.imshow("match", match)

    cv2.waitKey()
    cv2.destroyAllWindows()
