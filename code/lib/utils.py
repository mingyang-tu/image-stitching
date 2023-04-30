import numpy as np


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


def crop(img, threshold=10):
    mask = np.max(img, axis=2) == 0
    top, bot = 0, mask.shape[0] - 1
    for row in range(mask.shape[0]):
        if np.sum(mask[row, :]) <= threshold:
            top = row
            break
    for row in range(mask.shape[0]-1, -1, -1):
        if np.sum(mask[row, :]) <= threshold:
            bot = row
            break
    return img[top:bot+1, :, :]


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
