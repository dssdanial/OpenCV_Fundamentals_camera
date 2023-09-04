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

    # create BFMatcher object
    bf = cv.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(desc1,desc2,k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, live_camera, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imshow("Matching_SIFT",img3)
    # print(matches[0][0].distance)
    if cv.waitKey(5) & 0xFF ==ord('q'):
        break

Webcam.release()
cv.destroyAllWindows()