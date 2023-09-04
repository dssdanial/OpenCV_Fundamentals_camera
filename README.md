# OpenCV_Fundamentals_camera
Here in this repository we will be familiar with the concept of OpenCV functions and Camera Parameters





## 1-1- Brute-Force Matching with ORB Descriptors
We are using ORB descriptors to match features. Here, we will see a simple example on how to match features between two images. In this case, I have a **queryImage and a **trainImage**. We will try to find the queryImage in trainImage using feature matching. 
The first image is already taken, however, the second picture is a frame of live camera.

```python
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

What is this Matcher Object?

The result of matches = bf.match(des1,des2) line is a list of DMatch objects. This DMatch object has following attributes:

    DMatch.distance - Distance between descriptors. The lower, the better it is.
    DMatch.trainIdx - Index of the descriptor in train descriptors
    DMatch.queryIdx - Index of the descriptor in query descriptors
    DMatch.imgIdx - Index of the train image.

<br>

### why norm_hamming distance?

When comparing descriptors in computer vision, the Euclidian distance is usually understood as the square root of the sum of the squared differences between the two vectors' elements.
The ORB descriptors are vectors of binary values. If applying Euclidian distance to binary vectors, the squared result of a single comparison would always be 1 or 0, which is not informative when it comes to estimating the difference between the elements. The overall Euclidian distance would be the square root of the sum of those ones and zeroes, again not a good estimator of the difference between the vectors.
That's why the Hamming distance is used.

![image](https://github.com/dssdanial/OpenCV_Fundamentals_camera/assets/32397445/93766213-86c9-44e5-b1a6-72239ded50b4)


```python
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


## 1-2- Brute-Force Matching with SIFT Descriptors and Ratio Test

This time, we will use BFMatcher.knnMatch() to get k best matches. In this example, we will take k=2 so that we can apply ratio test explained by D.Lowe in his paper. 
Each keypoint of the first image is matched with a number of keypoints from the second image. We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement). Lowe's test checks that the two distances are sufficiently different. If they are not, then the keypoint is eliminated and will not be used for further calculations.
```
if distance1 < distance2 * a_constant then ....
```

Therefore,
```python
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

```

![BF_SIFT](https://github.com/dssdanial/OpenCV_Fundamentals_camera/assets/32397445/1911066d-6128-499d-a28a-a8818b93f71f)



## 2- FLANN based Matcher (Fast Library for Approximate Nearest Neighbors)

It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features. It works faster than BFMatcher for large datasets. 

For FLANN based matcher, we need to pass two dictionaries which specifies the algorithm to be used, its related parameters etc. 
**First dictionary** is IndexParams. For various algorithms, the information to be passed is explained in FLANN docs. As a summary, for algorithms like SIFT, SURF etc. you can pass following: 

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
```

**Second dictionary** is the SearchParams. It specifies the number of times the trees in the index should be recursively traversed. Higher values gives better precision, but also takes more time. If you want to change the value, pass search_params = dict(checks=100).

```python
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
```

**Result:**

![FLANN](https://github.com/dssdanial/OpenCV_Fundamentals_camera/assets/32397445/20856758-3d7b-466d-b1cf-21a88a962d0c)



