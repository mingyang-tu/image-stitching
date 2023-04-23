import numpy as np
import cv2
from lib import feature_match, image_match, linear_blend, e2e_alignment


if __name__ == "__main__":
    root = "../dataset/parrington/"
    names = [
        f"prtn{i:02d}.jpg" for i in range(18)
    ]

    images = []
    for i in names:
        images.append(cv2.imread(root + i))

    img_match = np.concatenate(images, axis=1)

    sift = cv2.SIFT_create()

    kps = []
    descs = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        print("Number of keypoints =", len(kp))
        kps.append([i.pt for i in kp])
        descs.append(des)

    lengths, offsets = feature_match(kps, descs)

    matching_tree = image_match(lengths, offsets)

    result = linear_blend(images, matching_tree)

    result = e2e_alignment(result, matching_tree, offsets)

    # cv2.imwrite("../result.jpg", result)

    cv2.imshow("Result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()
