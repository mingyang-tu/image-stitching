from .feature_match.match import feature_match
from .image_match import image_match
from .blend import linear_blend, get_linear_weight
from .alignment import e2e_alignment
from .wrap import cylindrical_projection
from .utils import crop

import cv2


def image_stitching(images, focal):
    images_cy = [cylindrical_projection(img, focal) for img in images]

    sift = cv2.SIFT_create()

    kps = []
    descs = []
    for img in images_cy:
        kp, des = sift.detectAndCompute(img, None)
        print("Number of keypoints =", len(kp))
        kps.append([i.pt for i in kp])
        descs.append(des)

    lengths, offsets = feature_match(kps, descs)

    matching_tree = image_match(lengths, offsets)

    img_weight = [
        get_linear_weight(img, focal) for img in images
    ]

    result = linear_blend(images_cy, matching_tree, img_weight)

    result = e2e_alignment(result, matching_tree, offsets)

    result = crop(result)

    return result
