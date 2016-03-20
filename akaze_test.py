import cv2
import numpy as np
from matplotlib import pyplot as plot

MIN_MATCH_COUNT = 10

#Load image and create grayscale
skridsko = cv2.imread('Images/skridsko/training/skridsko.jpg')
kub = cv2.imread('Images/skridsko/test/skridskokub.jpg')
graySko = cv2.cvtColor(skridsko,cv2.COLOR_BGR2GRAY)
grayKub = cv2.cvtColor(kub, cv2.COLOR_BGR2GRAY)


#Apply AKAZE
akaze = cv2.AKAZE_create()
kpS, desS = akaze.detectAndCompute(graySko,None)
kpK, desK = akaze.detectAndCompute(grayKub,None)


#Brute force matching
bfm = cv2.BFMatcher()
matches = bfm.knnMatch(desS, desK, k = 2)

#Ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)


if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kpS[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kpK[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M,mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h,w,d = skridsko.shape
    pts = np.float32( [ [0,0], [0,h-1], [w-1,h-1],[w-1,0] ] ).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    kub = cv2.polylines(kub,[np.int32(dst)],True, 255,3,cv2.LINE_AA)
else:
    print("Not enough matches")
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

img3 = cv2.drawMatches(skridsko,kpS,kub,kpK,good,None,**draw_params)

plot.imshow(img3), plot.show()

##cv2.imshow('Keypoints', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
