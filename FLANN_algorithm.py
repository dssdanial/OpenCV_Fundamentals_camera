#Brute-Force Matching with SIFT Descriptors and Ratio Test

import cv2 as cv
import numpy as np
import matplotlib as plt

img1=cv.imread("iphone.jpg")
Webcam=cv.VideoCapture(0)
# Initiate SIFT detector
sift = cv.SIFT_create()
kp1, desc1=sift.detectAndCompute(img1, None)

while(Webcam.isOpened()):

    ret, live_camera = Webcam.read()
    kp2, desc2 = sift.detectAndCompute(live_camera, None) # trainImage

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, kp1, live_camera, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

    cv.imshow("Matching_SIFT",img3)
    # print(matches[0][0].distance)
    if cv.waitKey(5) & 0xFF ==ord('q'):
        break

Webcam.release()
cv.destroyAllWindows()