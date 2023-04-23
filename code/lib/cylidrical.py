import numpy as np
import cv2 as cv


def cylindrical_projection(img , focal):
	h, w, c = img.shape
	cy_h = h
	cy_w = int(focal * np.arctan2(w/2, focal) * 2)
	cy_img = np.zeros((cy_h, cy_w, c))
	mask = np.zeros((cy_h, cy_w, c))

	for k in range(c):
		for cy_y in range(cy_h):
			for cy_x in range(cy_w):

				sft_cy_x = cy_x - cy_w/2
				sft_cy_y = cy_y - cy_h/2

				cor_x = np.tan(sft_cy_x / focal) * focal
				cor_y = sft_cy_y * np.sqrt(cor_x**2 + focal**2) / focal
				# print(cor_x, cor_y)
				cor_x += w/2
				cor_y += h/2

				if cor_x >= 0 and cor_x < w-1 and cor_y >= 0 and cor_y < h-1:
					# bilinear interpolation
					w00 = (cor_y - int(cor_y)) * (cor_x - int(cor_x))
					w01 = (cor_y - int(cor_y)) * (int(cor_x) + 1 - cor_x)
					w10 = (int(cor_y) + 1 - cor_y) * (cor_x - int(cor_x))
					w11 = (int(cor_y) + 1 - cor_y)  * (int(cor_x) + 1 - cor_x)

					cy_img[cy_y][cy_x][k] += w00 * img[int(cor_y)+1][int(cor_x)+1][k]
					cy_img[cy_y][cy_x][k] += w01 * img[int(cor_y)+1][int(cor_x)][k]
					cy_img[cy_y][cy_x][k] += w10 * img[int(cor_y)][int(cor_x)+1][k]
					cy_img[cy_y][cy_x][k] += w11 * img[int(cor_y)][int(cor_x)][k]
					mask[cy_y][cy_x] = 1
				# cy_img[cy_y][cy_x][k] = img[int(cor_y)][int(cor_x)-1][k]
				# mask[cy_y][cy_x] = 1
	return cy_img, mask

img = cv.imread("/tmp2/b07902058/DVE_hw2/memorial0064.png")

img, mask = cylindrical_projection(img , 700)

cv.imwrite("test.jpg", img)