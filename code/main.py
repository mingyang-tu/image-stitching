import numpy as np
import cv2
from lib import feature_match, display_matching, image_match, overlap


if __name__ == "__main__":
    root = "../test/"
    names = [
        "100-0038_img.jpg",
        "100-0023_img.jpg",
        "100-0024_img.jpg",
        "100-0039_img.jpg",
        "100-0040_img.jpg",
        "100-0025_img.jpg",
        "101-0104_img.jpg"
    ]

    images = []
    for i in names:
        images.append(cv2.imread(root + i))

    img_match = np.concatenate(images, axis=1)

    shifts = [0]
    for i in range(len(names)-1):
        shifts.append(shifts[-1] + images[i].shape[1])

    sift = cv2.SIFT_create()

    kps = []
    descs = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        kps.append([i.pt for i in kp])
        descs.append(des)

    pairs = feature_match(kps, descs)

    offsets = image_match(pairs)
    result = overlap(images, offsets)

    cv2.imshow("Result", result)

    display_matching(img_match, pairs, shifts)
