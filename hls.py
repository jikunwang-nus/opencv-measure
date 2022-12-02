import numpy as np
import cv2 as cv

if __name__ == '__main__':
    imgPath = './resources/table.jpg'
    img = cv.imread(imgPath)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(img_gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    # result用于标记角点，并不重要
    dst = cv.dilate(dst, None)
    # 最佳值的阈值，它可能因图像而异。
    print(dst.max())
    img[dst > 0.01 * dst.max()] = [255, 0, 0]
    cv.imshow('dst', img)
    cv.imwrite('hls.png', img_gray)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
