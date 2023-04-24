import cv2
from lib import image_stitching


if __name__ == "__main__":
    root = "../dataset/parrington/"
    names = [
        f"prtn{i:02d}.jpg" for i in range(18)
    ]

    images = []
    for i in names:
        images.append(cv2.imread(root + i))

    result = image_stitching(images, 704.916)

    # cv2.imwrite("../result.jpg", result)

    cv2.imshow("Result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()
