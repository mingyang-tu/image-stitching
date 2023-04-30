import numpy as np
import cv2

"""
    每張圖片的大小要一樣（或是可以改一下讓他可以不一樣）
"""


def linear_blend(images, matching_tree, img_weight):
    ROW, COL, _ = images[matching_tree.index].shape
    top, bot, left, right = maximum_offests(matching_tree)

    result = _linear_blend(images, matching_tree, img_weight, (ROW, COL), (top, bot, left, right))

    return result.astype(np.uint8)


def multi_band_blend(images, matching_tree, img_weight, band=5, sigma=5):
    ROW, COL, _ = images[matching_tree.index].shape
    top, bot, left, right = maximum_offests(matching_tree)

    w_max = get_max_weight(matching_tree, img_weight, (ROW, COL), (top, bot, left, right))

    masks = [(w > 0).astype(np.float64) for w in img_weight]

    result = np.zeros((ROW + top + bot, COL + left + right, 3), dtype=np.float64)

    last_I = [img.astype(np.float64) for img in images]
    curr_I = [img.copy() for img in last_I]
    last_W = [gaussian_blur_with_mask(w, masks[i], sigma) for i, w in enumerate(w_max)]
    curr_W = [w.copy() for w in last_W]
    for j in range(1, band):
        curr_B = []
        for i in range(len(img_weight)):
            curr_I[i] = gaussian_blur_with_mask(last_I[i], masks[i], sigma * (2 * j + 1) ** (1/2))
            curr_W[i] = gaussian_blur_with_mask(last_W[i], masks[i], sigma * (2 * j + 1) ** (1/2))
            curr_B.append(last_I[i] - curr_I[i])
        result += _linear_blend(curr_B, matching_tree, last_W, (ROW, COL), (top, bot, left, right))
        last_I, curr_I = curr_I, last_I
        last_W, curr_W = curr_W, last_W

    result += _linear_blend(last_I, matching_tree, last_W, (ROW, COL), (top, bot, left, right))

    return np.clip(result, 0, 255).astype(np.uint8)


def _linear_blend(images, matching_tree, img_weight, imsize, offsets):
    ROW, COL = imsize
    top, bot, left, right = offsets

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

    return result


def gaussian_blur_with_mask(image, mask, sigma):
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)

    blurred_mask[blurred_mask == 0] = 1
    if len(image.shape) == 2:
        return blurred_image / blurred_mask * mask
    else:
        result = np.zeros(image.shape, dtype=np.float64)
        for i in range(3):
            result[:, :, i] = blurred_image[:, :, i] / blurred_mask * mask
        return result


def get_max_weight(matching_tree, img_weight, imsize, offsets):
    ROW, COL = imsize
    top, bot, left, right = offsets
    ROW_r, COL_r = ROW + top + bot, COL + left + right

    w_idx = -np.ones((ROW_r, COL_r), dtype=np.int64)
    curr_max = np.zeros((ROW_r, COL_r), dtype=np.float64)

    def substitute_matrix(target, change_idx, change_val):
        target[change_idx] = change_val
        return target

    queue = [(matching_tree, (top, left))]
    while queue:
        node, (xi, yi) = queue.pop(0)
        weight = img_weight[node.index]
        changes = curr_max[xi:xi+ROW, yi:yi+COL] < weight

        w_idx[xi:xi+ROW, yi:yi+COL] = substitute_matrix(
            w_idx[xi:xi+ROW, yi:yi+COL],
            changes, node.index
        )

        curr_max[xi:xi+ROW, yi:yi+COL] = substitute_matrix(
            curr_max[xi:xi+ROW, yi:yi+COL],
            changes, weight[changes]
        )
        for child, (dy, dx) in node.children.items():
            queue.append((child, (xi+dx, yi+dy)))

    w_max = [np.zeros((ROW, COL), dtype=np.float64) for _ in range(np.max(w_idx)+1)]

    queue = [(matching_tree, (top, left))]
    while queue:
        node, (xi, yi) = queue.pop(0)
        w_max[node.index] = (w_idx[xi:xi+ROW, yi:yi+COL] == node.index).astype(np.float64)
        for child, (dy, dx) in node.children.items():
            queue.append((child, (xi+dx, yi+dy)))

    return w_max


def maximum_offests(matching_tree):

    def _dfs(node, curr_x, curr_y):
        offsets[0], offsets[1] = min(curr_x, offsets[0]), max(curr_x, offsets[1])
        offsets[2], offsets[3] = min(curr_y, offsets[2]), max(curr_y, offsets[3])
        for child, (y, x) in node.children.items():
            _dfs(child, curr_x+x, curr_y+y)

    offsets = [0, 0, 0, 0]  # top, bot, left, right
    _dfs(matching_tree, 0, 0)

    return -offsets[0], offsets[1], -offsets[2], offsets[3]
