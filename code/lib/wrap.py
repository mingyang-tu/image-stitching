import numpy as np
from .utils import bilinear_interpolation


def cylindrical_projection(img, focal):
    ROW, COL, CHANNEL = img.shape

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
    img_pad = np.pad(img, (pad_x, pad_y, (0, 0)))

    img_cy = np.zeros((CY_ROW, CY_COL, CHANNEL), dtype=np.uint8)

    for i in range(CHANNEL):
        img_cy[:, :, i] = np.clip(
            bilinear_interpolation(img_pad[:, :, i], m1 + pad_x[0], n1 + pad_y[0]),
            0, 255
        ).astype(np.uint8)

    return img_cy
