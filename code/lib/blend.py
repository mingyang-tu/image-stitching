import numpy as np
import cv2
from math import ceil


def overlap(images, offsets):
    top, bot, left, right = 0, 0, 0, 0
    for (x, y) in offsets:
        if x > 0:
            right += x
        else:
            left -= x
        if y > 0:
            bot += y
        else:
            top -= y

    top, bot, left, right = ceil(top), ceil(bot), ceil(left), ceil(right)
    padded = [np.pad(img, ((top, bot), (left, right), (0, 0))) for img in images]

    acc_x, acc_y = 0, 0
    last = padded[0]
    for i in range(1, len(padded)):
        acc_x += offsets[i-1][0]
        acc_y += offsets[i-1][1]
        curr = shift(padded[i], (acc_x, acc_y))
        curr[curr == 0] = last[curr == 0]
        last = curr
    return last


def shift(img, offset):
    translation_matrix = np.array(
        [[1, 0, offset[0]], [0, 1, offset[1]]],
        dtype=np.float64
    )
    img_shift = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    return img_shift
