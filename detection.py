import cv2
import numpy as np
from matplotlib import pyplot as plot
from os import listdir
from os.path import isfile, join
import copy
from class_sift import Sift
from class_akaze import AKaze

MIN_MATCH_COUNT = 10
FOLDER1 = 'Images/skridsko/'
FOLDER2 = 'Images/panter/'


def main():
    #Ask wich object we are running tests on
    choice = raw_input('choose folder skidsko (s) or panter (p): ')

    if choice == 's':
        folder = FOLDER1
    else:
        folder = FOLDER2
    tr_names, test_names = read_folders(folder)

    print('Available training images: \n')
    print(tr_names)
    tr = raw_input('\n choose training picture: ')
    print('\n Available test images: \n')
    print(test_names)
    test = raw_input('\n choose test picture: ')

    img1 =  folder + 'training/' + tr +'.jpg'
    print('using training image: ' + img1)
    img2 = folder + 'test/' + test +'.jpg'
    print('using test image: ' + img2)
    tr_img, test_img, tr_grey, test_grey = load__greyScale(img1, img2)

    sift = Sift()
    kpAS, descAS, kpBS, descBS, good_matches_sift = sift.match(tr_grey, test_grey)
    akaze = AKaze()
    kpAK, descAK, kpBK, descBK, good_matches_akaze = akaze.match(tr_grey, test_grey)

    maskS, dtsS = location_extraction(kpAS, kpBS, good_matches_sift,tr_img)
    maskK, dtsK = location_extraction(kpAK, kpBK, good_matches_akaze,tr_img)

    if dtsS !=None and dtsK != None:

        #Copy of image so that lines from first method do not last to second.
        tmp_img = copy.copy(test_img)
        res_sift_img = create_results(tr_img, test_img, kpAS, kpBS, dtsS,
                                            maskS, good_matches_sift)
        res_akaze_img = create_results(tr_img, tmp_img, kpAK, kpBK, dtsK,
                                            maskK, good_matches_akaze)

        print("jek!!!")
        plot.subplot(211), plot.imshow(res_sift_img), plot.title('Sift')
        plot.subplot(212), plot.imshow(res_akaze_img), plot.title('AKaze')

        fig_name = 'results/' +tr + '_vs_' + test +'.png'
        plot.savefig(fig_name)
        plot.show()



    elif dtsK != None:
        print('Sift misslyckades')

        res_akaze_img = create_results(tr_img, tmp_img, kpAK, kpBK, dtsK,
                                            maskK, good_matches_akaze)


        plot.imshow(res_akaze_img), plot.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif dtsS != None:
        print('akaze misslyckades')
        res_sift_img = create_results(tr_img, test_img, kpAS, kpBS, dtsS,
                                            maskS, good_matches_sift)

        plot.imshow(res_sift_img), plot.show()
    else:
        print('Both methods could not find a match')

        cv2.waitKey(0)
        cv2.destroyAllWindows()


#Read a folder containing images and return a list of urls
def read_folders(folder):

    tr_path = folder + 'training'
    #List of named of all training images
    tr_names = [f for f in listdir(tr_path) if isfile(join(tr_path, f))]

    #List of all test images
    test_path = folder + 'test'
    test_names = [f for f in listdir(test_path) if isfile(join(test_path, f))]

    return tr_names, test_names


#Read an image and return its grey picture
def load__greyScale(train, test):
    imgA = cv2.imread(train)
    imgB = cv2.imread(test)

    greyA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    greyB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    return imgA, imgB, greyA, greyB





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
    draw_params = dict(singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)
    res_img = cv2.drawMatches(tr_img,kpA,tmp_img,kpB,good_matches,None,**draw_params)

    return res_img

main()
