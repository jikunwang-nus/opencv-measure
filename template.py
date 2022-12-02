import cv2 as cv


def create_template():
    file = "resources/afterEdges.png"
    img = cv.imread(file)
    temp = img[2632:2749, 2077:2205]
    cv.imshow('temp', temp)
    cv.imwrite('template.png', temp)
