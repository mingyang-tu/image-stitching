import numpy as np


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
