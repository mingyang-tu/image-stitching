import numpy as np


def e2e_alignment(result, matching_tree, offsets):
    left, right = [0, 0], [0, 0]
    idx_l = idx_r = matching_tree.index
    queue = [(matching_tree, (0, 0))]
    while queue:
        node, (curr_x, curr_y) = queue.pop(0)
        if curr_y < left[1]:
            left[0], left[1] = curr_x, curr_y
            idx_l = node.index
        if curr_y > right[1]:
            right[0], right[1] = curr_x, curr_y
            idx_r = node.index
        for child, (dy, dx) in node.children.items():
            queue.append((child, (curr_x+dx, curr_y+dy)))

    diff_x = right[0] - left[0] + offsets[idx_r][idx_l][1]
    diff_y = right[1] - left[1] + offsets[idx_r][idx_l][0]

    if 10 < abs(diff_y) < float("inf"):
        print(f"Drift = {diff_x}")
        rotation = np.array([
            [1, -diff_x / diff_y], [0, 1]
        ], dtype=np.float64)
        return affine_transformation(result, rotation)
    else:
        return result


def bilinear_interpolation(img, m1, n1):
    m0 = np.clip(np.floor(m1), 0, img.shape[0] - 2).astype(int)
    n0 = np.clip(np.floor(n1), 0, img.shape[1] - 2).astype(int)

    a = m1 - m0
    b = n1 - n0

    return (
        (1 - a) * (1 - b) * img[m0, n0] +
        a * (1 - b) * img[m0 + 1, n0] +
        (1 - a) * b * img[m0, n0 + 1] +
        a * b * img[m0 + 1, n0 + 1]
    )


def inv_2x2_mat(mat):
    assert mat.shape[0] == 2 and mat.shape[1] == 2
    (a, b), (c, d) = mat
    det = a * d - b * c
    assert det != 0
    return 1 / det * np.array([[d, -b], [-c, a]], dtype=np.float64)


def affine_transformation(img, mat):
    ROW, COL, CHANNEL = img.shape
    center = np.array([ROW // 2, COL // 2], dtype=np.float64).reshape(-1, 1, 1)

    row_i, col_i = np.mgrid[0: ROW, 0: COL].astype(np.float64) - center

    (ai, bi), (ci, di) = inv_2x2_mat(mat)
    m1 = ai * row_i + bi * col_i + center[0]
    n1 = ci * row_i + di * col_i + center[1]

    m1_floor = np.floor(m1).astype(int)
    n1_floor = np.floor(n1).astype(int)

    pad_x = max(-np.min(m1_floor), 0), max(np.max(m1_floor) - ROW + 1, 0)
    pad_y = max(-np.min(n1_floor), 0), max(np.max(n1_floor) - COL + 1, 0)
    img_pad = np.pad(img, (pad_x, pad_y, (0, 0)))

    img_new = np.zeros(img.shape, dtype=np.uint8)
    for i in range(CHANNEL):
        img_new[:, :, i] = np.clip(
            bilinear_interpolation(img_pad[:, :, i], m1 + pad_x[0], n1 + pad_y[0]),
            0, 255
        ).astype(np.uint8)

    return img_new
