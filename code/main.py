import cv2
from lib import image_stitching


if __name__ == "__main__":
    root = "../image/"
    names = [
        f"img{i}.JPG" for i in range(9)
    ]

    images = []
    for i in names:
        images.append(cv2.resize(cv2.imread(root + i), (0, 0), fx=1/4, fy=1/4))

    result = image_stitching(images, 800)

    cv2.imwrite("../result.png", result)

    cv2.imshow("Result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()
