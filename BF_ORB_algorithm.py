import cv2 as cv
import numpy as np
import matplotlib as plt

img1=cv.imread("iphone.jpg")
# rec_video=cv.VideoCapture("30Seconds.mp4")
Webcam=cv.VideoCapture(0)

orb=cv.ORB_create(nfeatures=1000)
kp1, desc1=orb.detectAndCompute(img1, None)
# img1=cv.drawKeypoints(img1, kp1, None) # queryImage

print(desc1)
while(Webcam.isOpened()):

    ret, live_camera = Webcam.read()
    kp2, desc2 = orb.detectAndCompute(live_camera, None) # trainImage

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(desc1, desc2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance) #key (Optional)- A function that serves as a key for the sort comparison.
    # Draw first 10 matches. best matches (with low distance) come to front.
    img3 = cv.drawMatches(img1, kp1, live_camera, kp2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #live_camera = cv.drawKeypoints(live_camera, kp2, None)
    #ret2, video=rec_video.read()
    #height, width, channel=img1.shape
    #live_camera=cv.resize(live_camera, (width, height))
    #video=cv.resize(video,(width, height) )

    cv.imshow("matching_ORB",img3)
    #plots
    # cv.imshow("live", live_camera)
    # cv.imshow("image", img1)
    cv.waitKey(1)
    # cv.imshow("recorded video", video)



    if cv.waitKey(5) & 0xFF ==ord('q'):
        break
