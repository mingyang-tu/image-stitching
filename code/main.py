import numpy as np
import cv2
from lib import feature_match, image_match, linear_blend


if __name__ == "__main__":
    root = "../dataset/ntulib/"
    names = [
        f"img{i}.JPG" for i in range(1, 9)
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
        print("Number of keypoints =", len(kp))
        kps.append([i.pt for i in kp])
        descs.append(des)

    pairs = feature_match(kps, descs)

    matching_tree = image_match(pairs)

    result = linear_blend(images, matching_tree)

    # cv2.imwrite("../result.jpg", result)

    cv2.imshow("Result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()
