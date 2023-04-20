import numpy as np


def linear_blend(images, matching_tree):
    ROW, COL, _ = images[matching_tree.index].shape
    top, bot, left, right = maximum_offests(matching_tree)

    result = np.zeros((ROW + top + bot, COL + left + right, 3), dtype=np.float64)
    weights = np.zeros((ROW + top + bot, COL + left + right), dtype=np.float64)

    queue = [(matching_tree, (top, left))]
    while queue:
        node, (xi, yi) = queue.pop(0)
        weight = get_linear_weight(images[node.index])
        weights[xi:xi+ROW, yi:yi+COL] += weight
        for i in range(3):
            result[xi:xi+ROW, yi:yi+COL, i] += weight * images[node.index][:, :, i]
        for child, (dy, dx) in node.children.items():
            queue.append((child, (xi+dx, yi+dy)))

    weights[weights == 0] = 1
    for i in range(3):
        result[:, :, i] /= weights

    return result.astype(np.uint8)


def get_linear_weight(img):
    binary = np.array(np.max(img, axis=2) > 0, dtype=np.float64)

    ROW, COL = binary.shape
    c_x, c_y = (ROW - 1) // 2, (COL - 1) // 2

    w_x = np.concatenate([np.arange(0, c_x+1), np.arange(ROW-c_x-2, -1, -1)])
    w_x = w_x / np.max(w_x)
    w_y = np.concatenate([np.arange(0, c_y+1), np.arange(COL-c_y-2, -1, -1)])
    w_y = w_y / np.max(w_y)

    w_xy = np.dot(w_x.reshape(-1, 1), w_y.reshape(1, -1))

    return w_xy


def maximum_offests(matching_tree):

    def _dfs(node, curr_x, curr_y):
        offsets[0], offsets[1] = min(curr_x, offsets[0]), max(curr_x, offsets[1])
        offsets[2], offsets[3] = min(curr_y, offsets[2]), max(curr_y, offsets[3])
        for child, (y, x) in node.children.items():
            _dfs(child, curr_x+x, curr_y+y)

    offsets = [0, 0, 0, 0]  # top, bot, left, right
    _dfs(matching_tree, 0, 0)

    return -offsets[0], offsets[1], -offsets[2], offsets[3]

