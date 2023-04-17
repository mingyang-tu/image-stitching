import numpy as np
import cv2
from math import ceil


def overlap(images, matching_tree):
    top, bot, left, right = 0, 0, 0, 0

    Q = [matching_tree]
    while Q:
        curr = Q.pop(0)
        for child, (x, y) in curr.children.items():
            if x > 0:
                right += x
            else:
                left -= x
            if y > 0:
                bot += y
            else:
                top -= y
            Q.append(child)

    top, bot, left, right = ceil(top), ceil(bot), ceil(left), ceil(right)
    padded = [np.pad(img, ((top, bot), (left, right), (0, 0))) for img in images]

    def _dfs(node, sum_x=0, sum_y=0):
        curr = shift(padded[node.index], (sum_x, sum_y))
        for child, (x, y) in node.children.items():
            shift_child = _dfs(child, sum_x+x, sum_y+y)
            curr[curr == 0] = shift_child[curr == 0]
        return curr

    return _dfs(matching_tree)


def shift(img, offset):
    translation_matrix = np.array(
        [[1, 0, offset[0]], [0, 1, offset[1]]],
        dtype=np.float64
    )
    img_shift = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    return img_shift
