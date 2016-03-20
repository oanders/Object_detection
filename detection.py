import cv2
import numpy as np
from matplotlib import pyplot as plot
from os import listdir
from os.path import isfile, join
import copy

MIN_MATCH_COUNT = 10



def main():
    tr1, test1, tr2, test2 = read_folders()
    img1 = tr1[0]
    print('using training image: ' + img1)
    img2 = test1[2]
    print('using test image: ' + img2)
    tr_img, test_img, tr_grey, test_grey = load__greyScale(img1, img2)

    sift = Sift()
    kpAS, descAS, kpBS, descBS = sift.key_desc(tr_grey, test_grey)
    akaze = AKaze()
    kpAK, descAK, kpBK, descBK = akaze.key_desc(tr_grey, test_grey)

    good_matches_sift = bf_matching(descAS, descBS)
    good_matches_akaze = bf_matching(descAK, descBK)


    maskS, dtsS = location_extraction(kpAS, kpBS, good_matches_sift,tr_img)
    maskK, dtsK = location_extraction(kpAK, kpBK, good_matches_akaze,tr_img)

    if dtsS !=None and dtsK != None:

        #Copy of image so that lines from first method do not last to second.
        tmp_img = copy.copy(test_img)
        res_sift_img = create_results(tr_img, test_img, kpAS, kpBS, dtsS,
                                            maskS, good_matches_sift)
        res_akaze_img = create_results(tr_img, tmp_img, kpAK, kpBK, dtsK,
                                            maskK, good_matches_akaze)

        plot.subplot(211), plot.imshow(res_sift_img), plot.title('Sift')
        plot.subplot(212), plot.imshow(res_akaze_img), plot.title('AKaze')

        plot.show()

    elif dtsS == None:
        print('Sift misslyckades')

        res_akaze_img = create_results(tr_img, tmp_img, kpAK, kpBK, dtsK,
                                            maskK, good_matches_akaze)


        plot.imshow(res_akaze_img), plot.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif dtsK == None:
        print('akaze misslyckades')
        res_sift_img = create_results(tr_img, test_img, kpAS, kpBS, dtsS,
                                            maskS, good_matches_sift)

        plot.imshow(res_sift_img), plot.show()


        cv2.waitKey(0)
        cv2.destroyAllWindows()


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

    return imgA, imgB, greyA, greyB

#A class that calls for openCV function Sift
class Sift:
    #Contructor for class, objects of class SIFT
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

    #Function that detects keypoints in image and creates a
    #descriptor for each one
    def key_desc(self, grey1, grey2):
        word = None
        kpA, descA = self.sift.detectAndCompute(grey1, word)
        kpB, descB = self.sift.detectAndCompute(grey2, word)

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
        kpA, descA = self.akaze.detectAndCompute(grey1, word)
        kpB, descB = self.akaze.detectAndCompute(grey2, word)

        return kpA, descA, kpB, descB

#Uses brute force matching
def bf_matching(descA, descB):
    bfm = cv2.BFMatcher()
    matches = bfm.knnMatch(descA, descB, k = 2)

    #Ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    return good_matches

def location_extraction(kpA, kpB, good_matches, tr_img):
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpA[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kpB[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        M,mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w,d = tr_img.shape
        pts = np.float32( [ [0,0], [0,h-1], [w-1,h-1],[w-1,0] ] ).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        return matchesMask, dst

    else:
        print("Not enough matches")
        matchesMask = None
        return matchesMask, None

#
def create_results(tr_img, tmp_img, kpA, kpB, dst, matchesMask, good_matches):
    tmp_img = cv2.polylines(tmp_img,[np.int32(dst)],True, 255,3,cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)
    res_img = cv2.drawMatches(tr_img,kpA,tmp_img,kpB,good_matches,None,**draw_params)

    return res_img

main()
