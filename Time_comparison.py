import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
import seaborn as sb

Webcam=cv.VideoCapture(0)
sift = cv.SIFT_create()# Initiate SIFT detector
# surf = cv.xfeatures2d.SIFT_create()
total_time=0
total_features=0
counter=0

# while(Webcam.isOpened()):
#
#     ret, live_camera = Webcam.read()
#     start_time = time.time()
#     kp2, desc2 = sift.detectAndCompute(live_camera, None) # trainImage
#     total_time+=time.time()-start_time
#     total_features+=np.array(kp2).shape[0]
#     cv.imshow("matching_ORB",live_camera)
#
#     counter+=1
#     if counter>50:
#         break
#
#
#     if cv.waitKey(5) & 0xFF ==ord('q'):
#         break
#
#
# sift_time=total_time/5
# sift_features = total_features/5
# print("Average time for SIFT features: ",sift_time)
# print("Average number of SIFT features: ",sift_features)
#


orb=cv.ORB_create(nfeatures=1000)
while(Webcam.isOpened()):

    ret, live_camera = Webcam.read()
    start_time=time.time()
    kp,des = orb.detectAndCompute(live_camera,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
    cv.imshow("matching_ORB", live_camera)

    counter+=1
    if counter>50:
        break


    if cv.waitKey(5) & 0xFF ==ord('q'):
        break

orb_time=total_time/5
orb_features = total_features/5

print("Average time for ORB features: ",orb_time)
print("Average number of ORB features: ",orb_features)








Webcam.release()
cv.destroyAllWindows()




#
# fig = plt.figure()
# sns.set()
# ax = fig.add_axes([0,0,1,1])
# methods = ['SIFT', 'SURF', 'ORB']
# times = [sift_time*1000,surf_time*1000,orb_time*1000]
# ax.barh(methods,times,color=('green','blue','orange'))
# ax.set_ylabel('Feature Extractor')
# ax.set_xlabel('Time (ms)')
# ax.set_title('Average time to compute ~300 Key-Point Descriptors')
# plt.show()
