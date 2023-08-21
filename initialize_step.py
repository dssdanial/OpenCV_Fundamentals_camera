import cv2 as cv
import numpy as np
import matplotlib as plt

img=cv.imread("iphone.jpg")
rec_video=cv.VideoCapture("30Seconds.mp4")
Webcam=cv.VideoCapture(5)

orb=cv.ORB_create(nfeatures=1000)
kp, desc=orb.detectAndCompute(img, None)
img=cv.drawKeypoints(img, kp, None)

print(desc)
while(Webcam.isOpened()):

    ret, live_video = Webcam.read()
    kp2, desc2 = orb.detectAndCompute(live_video, None)
    live_video = cv.drawKeypoints(live_video, kp2, None)

    ret2, video=rec_video.read()
    height, width, channel=img.shape
    live_video=cv.resize(live_video, (width, height))
    video=cv.resize(video,(width, height) )
    cv.imshow("live", live_video)
    cv.imshow("image", img)
    cv.waitKey(1)
    # cv.imshow("recorded video", video)



    if cv.waitKey(5) & 0xFF ==ord('q'):
        break
