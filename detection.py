import cv2
import numpy as np
from matplotlib import pyplot as plot
from os import listdir
from os.path import isfile, join

MIN_MATCH_COUNT = 10



def main():
    tr1, test1, tr2, test2 = read_folders()
    img1 = tr1[0]
    img2 = test1[0]
    grey1, grey2 = load__greyScale(img1, img2)


#Read a folder containing images and return a list of urls
def read_folders():
    #------------------SKIDSKO--------------------------
    #Path for the training images of skidsko
    mps = 'Images/skridsko/training'
    #List of named of all training images
    tr_s = [mps + '/' + f for f in listdir(mps) if isfile(join(mps, f))]

    #PAth for test images of skrisko
    tps = 'Images/skridsko/test'
    #List of all test images
    test_s = [tps + '/' + f for f in listdir(tps) if isfile(join(tps, f))]

    #---------PANTER------------------
    #Path for the training images of skidsko
    mpp = 'Images/panter/training'
    #List of named of all training images
    tr_p = [mpp + '/' + f for f in listdir(mpp) if isfile(join(mpp, f))]

    #PAth for test images of skrisko
    tpp = 'Images/panter/test'
    #List of all test images
    test_p = [tpp + '/'+ f for f in listdir(tpp) if isfile(join(tpp, f))]

    return tr_s, test_s, tr_p, test_p

#Read an image and return its grey picture
def load__greyScale(train, test):
    imgA = cv2.imread(train)
    imgB = cv2.imread(test)

    greyA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    greyB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    return greyA, greyB

#A class that calls for openCV function Sift
class Sift:
    #Contructor for class, objects of class SIFT
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

    #Function that detects keypoints in image and creates a
    #descriptor for each one
    def key_desc(self, grey1, grey2):
        word = None
        kpA, descA = sift.detectAndCompute(grey1, word)
        kpB, descB = sift.detectAndCompute(grey2, word)

        return kpA, descA, kpB, descB

#Class that calls for openCV function akaze
class AKaze:
    #Constructor for objects of class AKaze
    def __init__(self):
        self.akaze = cv2.AKAZE_create()

    #Function that detects keypoints in image and creates a
    #descriptor for each one
    def key_desc(self, grey1, grey2):
        word = None
        kpA, descA = akaze.detectAndCompute(grey1, word)
        kpB, descB = akaze.detectAndCompute(grey2, word)

        return kpA, descA, kpB, descB

#Uses brute force matching
def bf_matching(descA, decB):
    bfm = cv2.BFMatcher()
    matches = bfm.knnMatch(descA, descB, k = 2)

    #Ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    return good

def location_extraction(keyA, keyB, good, tr_Img, test_Img):
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M,mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w,d = skridsko.shape
        pts = np.float32( [ [0,0], [0,h-1], [w-1,h-1],[w-1,0] ] ).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

    else:
        print("Not enough matches")
        matchesMask = None

    return matchesMask, pts, dst    
def draw_results():
    kub = cv2.polylines(kub,[np.int32(dst)],True, 255,3,cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)
    img3 = cv2.drawMatches(skridsko,kpS,kub,kpK,good,None,**draw_params)

main()
