import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
if __name__ == '__main__':
    img = cv.imread('/Users/jeff/Documents/pic/table.jpg', 0)
    # 用默认值初始化FAST对象
    fast = cv.FastFeatureDetector_create()
    # 寻找并绘制关键点
    kp = fast.detect(img, None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    # 打印所有默认参数
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    cv.imwrite('fast_true.png', img2)
    # 关闭非极大抑制
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
    img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    cv.imwrite('fast_false.png', img3)
