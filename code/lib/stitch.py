from .SIFT import SIFT
from .feature_match.match import feature_match
from .image_match import image_match
from .blend import linear_blend, get_linear_weight
from .alignment import e2e_alignment
from .wrap import cylindrical_projection
from .utils import crop


def image_stitching(images, focal):
    images_cy = [cylindrical_projection(img, focal) for img in images]

    sift = SIFT()

    kps = []
    descs = []
    for img in images_cy:
        kp, des = sift.fit(img)
        kps.append([(i.x, i.y) for i in kp])
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
