import numpy as np
from .utils import bilinear_interpolation


def linear_blend(images, matching_tree, img_weight):
    """
    Todo:
    每張圖片的大小要一樣（或是可以改一下讓他可以不一樣）
    """
    ROW, COL, _ = images[matching_tree.index].shape
    top, bot, left, right = maximum_offests(matching_tree)

    ROW_r, COL_r = ROW + top + bot, COL + left + right
    result = np.zeros((ROW_r, COL_r, 3), dtype=np.float64)
    weights = np.zeros((ROW_r, COL_r), dtype=np.float64)

    queue = [(matching_tree, (top, left))]
    while queue:
        node, (xi, yi) = queue.pop(0)
        weight = img_weight[node.index]
        weights[xi:xi+ROW, yi:yi+COL] += weight
        for i in range(3):
            result[xi:xi+ROW, yi:yi+COL, i] += weight * images[node.index][:, :, i]
        for child, (dy, dx) in node.children.items():
            queue.append((child, (xi+dx, yi+dy)))

    weights[weights == 0] = 1
    for i in range(3):
        result[:, :, i] /= weights

    return result.astype(np.uint8)


def get_linear_weight(img, focal):
    ROW, COL, _ = img.shape
    c_x, c_y = (ROW - 1) // 2, (COL - 1) // 2

    w_x = np.concatenate([np.arange(0, c_x+1), np.arange(ROW-c_x-2, -1, -1)])
    w_x = w_x / np.max(w_x)
    w_y = np.concatenate([np.arange(0, c_y+1), np.arange(COL-c_y-2, -1, -1)])
    w_y = w_y / np.max(w_y)

    w_xy = np.dot(w_x.reshape(-1, 1), w_y.reshape(1, -1))

    return project_weight(w_xy, focal)


def project_weight(weight, focal):
    ROW, COL = weight.shape

    CY_ROW = ROW
    CY_COL = int(focal * np.arctan(COL / 2 / focal) * 2)

    center = np.array([CY_ROW / 2, CY_COL / 2], dtype=np.float64).reshape(-1, 1, 1)
    row_i, col_i = np.mgrid[0: CY_ROW, 0: CY_COL].astype(np.float64) - center

    n1 = np.tan(col_i / focal) * focal
    m1 = row_i * np.sqrt(n1**2 + focal**2) / focal

    m1 += ROW / 2
    n1 += COL / 2

    m1_floor = np.floor(m1).astype(int)
    n1_floor = np.floor(n1).astype(int)

    pad_x = max(-np.min(m1_floor), 0), max(np.max(m1_floor) - ROW + 1, 0)
    pad_y = max(-np.min(n1_floor), 0), max(np.max(n1_floor) - COL + 1, 0)
    img_pad = np.pad(weight, (pad_x, pad_y))

    img_cy = np.clip(
        bilinear_interpolation(img_pad, m1 + pad_x[0], n1 + pad_y[0]),
        0, 1
    )

    return img_cy


def maximum_offests(matching_tree):

    def _dfs(node, curr_x, curr_y):
        offsets[0], offsets[1] = min(curr_x, offsets[0]), max(curr_x, offsets[1])
        offsets[2], offsets[3] = min(curr_y, offsets[2]), max(curr_y, offsets[3])
        for child, (y, x) in node.children.items():
            _dfs(child, curr_x+x, curr_y+y)

    offsets = [0, 0, 0, 0]  # top, bot, left, right
    _dfs(matching_tree, 0, 0)

    return -offsets[0], offsets[1], -offsets[2], offsets[3]
