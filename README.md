# OpenCV_Fundamentals_camera
Here in this repository we will be familiar with the concept of OpenCV functions and Camera Parameters





## Brute-Force Matching with ORB Descriptors
We are using ORB descriptors to match features. Here, we will see a simple example on how to match features between two images. In this case, I have a **queryImage and a **trainImage**. We will try to find the queryImage in trainImage using feature matching. 
The first image is already taken, however, the second picture is a frame of live camera.

```
import cv2 as cv
import numpy as np
import matplotlib as plt

img1=cv.imread("iphone.jpg")
rec_video=cv.VideoCapture("30Seconds.mp4")
Webcam=cv.VideoCapture(0)

orb=cv.ORB_create(nfeatures=1000)
kp1, desc1=orb.detectAndCompute(img1, None)
img1=cv.drawKeypoints(img1, kp1, None) # queryImage

```
Next we create a BFMatcher object with distance measurement `cv.NORM_HAMMING` (since we are using __ORB__) and **crossCheck** is switched on for better results. Then we use `Matcher.match()` method to get the best matches in two images. 
We sort them in **ascending order of their distances** so that best matches (with low distance) come to front. 

<br>

### why norm_hamming distance?

When comparing descriptors in computer vision, the Euclidian distance is usually understood as the square root of the sum of the squared differences between the two vectors' elements.
The ORB descriptors are vectors of binary values. If applying Euclidian distance to binary vectors, the squared result of a single comparison would always be 1 or 0, which is not informative when it comes to estimating the difference between the elements. The overall Euclidian distance would be the square root of the sum of those ones and zeroes, again not a good estimator of the difference between the vectors.
That's why the Hamming distance is used.

![image](https://github.com/dssdanial/OpenCV_Fundamentals_camera/assets/32397445/93766213-86c9-44e5-b1a6-72239ded50b4)


```
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

```
Result with 20 best matchers
#### Standard view
![001](https://github.com/dssdanial/OpenCV_Fundamentals_camera/assets/32397445/3b846149-a373-4a01-82ae-2e86cf67f328)


#### Rotated view
![002](https://github.com/dssdanial/OpenCV_Fundamentals_camera/assets/32397445/37a2a2dc-27be-4cc4-8618-4e21e4f31834)



https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
