import numpy as np
import cv2


def linear_blend(images, matching_tree):
    top, bot, left, right = maximum_offests(matching_tree)

    padded = [np.pad(img, ((top, bot), (left, right), (0, 0))).astype(np.float64) for img in images]

    ROW, COL, _ = padded[0].shape

    weights = np.zeros((ROW, COL), dtype=np.float64)
    weighted_sum = np.zeros((ROW, COL, 3), dtype=np.float64)

    queue = [(matching_tree, (0., 0.))]
    while queue:
        node, (sum_x, sum_y) = queue.pop(0)
        curr = shift_image(padded[node.index], (sum_x, sum_y))
        weight = get_linear_weight(curr)
        weights += weight
        for i in range(3):
            weighted_sum[:, :, i] += weight * curr[:, :, i]
        for child, (x, y) in node.children.items():
            queue.append((child, (sum_x+x, sum_y+y)))

    weights[weights == 0] = 1
    for i in range(3):
        weighted_sum[:, :, i] /= weights

    return weighted_sum.astype(np.uint8)


def get_linear_weight(img):
    binary = np.array(np.mean(img, axis=2) > 0, dtype=np.float64)

    top, bot, left, right = find_corner(binary)
    c_x, c_y = (top + bot) // 2, (left + right) // 2

    w_x = np.concatenate([np.arange(0, c_x-top+1), np.arange(bot-c_x-1, -1, -1)])
    w_x = w_x / np.max(w_x)
    weight_x = np.zeros(binary.shape, dtype=np.float64)
    weight_x[top:bot+1, left:right+1] = np.tile(w_x, (int(right-left+1), 1)).T

    w_y = np.concatenate([np.arange(0, c_y-left+1), np.arange(right-c_y-1, -1, -1)])
    w_y = w_y / np.max(w_y)
    weight_y = np.zeros(binary.shape, dtype=np.float64)
    weight_y[top:bot+1, left:right+1] = np.tile(w_y, (int(bot-top+1), 1))

    weight = weight_x * weight_y
    weight /= np.max(weight)

    return weight


def find_corner(img):
    x_range, y_range = np.nonzero(img)
    return np.min(x_range), np.max(x_range), np.min(y_range), np.max(y_range)


def maximum_offests(matching_tree):

    def _dfs(node, curr_x, curr_y):
        offsets[0], offsets[1] = min(curr_y, offsets[0]), max(curr_y, offsets[1])
        offsets[2], offsets[3] = min(curr_x, offsets[2]), max(curr_x, offsets[3])
        for child, (x, y) in node.children.items():
            _dfs(child, curr_x+x, curr_y+y)

    offsets = [0, 0, 0, 0]  # top, bot, left, right
    _dfs(matching_tree, 0, 0)

    return -offsets[0], offsets[1], -offsets[2], offsets[3]


def shift_image(img, offset):
    translation_matrix = np.array(
        [[1, 0, offset[0]], [0, 1, offset[1]]],
        dtype=np.float64
    )
    img_shift = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    return img_shift
