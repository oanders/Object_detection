import cv2
import numpy as np
from matplotlib import pyplot as plot
import os
#from os import listdir, makedirs
#from os.path import isfile, join, isdir
import copy
from class_sift import Sift
from class_akaze import AKaze
from class_orb import ORB


MIN_MATCH_COUNT = 10
FOLDER1 = 'Images/skridsko/'
FOLDER2 = 'Images/panter/'
FOLDER3 = 'Images/flaska/'
FOLDER4 = 'Images/kontroll/'


def main():
    #Ask wich object we are running tests on
    choice = raw_input('choose folder skidsko , panter, flaska, kontroll: ')

    if choice == 'skridsko':
        folder = FOLDER1
    elif choice == 'panter':
        folder = FOLDER2
    elif choice == 'flaska':
        folder = FOLDER3
    elif choice == 'kontroll':
        folder = FOLDER4
    else:
        print('Invalid choice')
        return main()
    names = read_folders(folder)

    print('Available test: \n')
    print(names)
    choice2 = raw_input('\n choose test: ')
    directory = 'results/' + choice + '/' + choice2


    img1 =  folder + choice2 + '/' + 'tr.jpg'
    print('using training image: ' + img1)
    tr_img, tr_grey = load__greyScale(img1)
    #Loop through test images
    nr = read_nr_images(folder + choice2)
    i = 1
    while i < nr:
        img2 = folder + choice2 + '/' + 't' + str(i) + '.jpg'
        print('using test image: ' + img2)
        test_img, test_grey = load__greyScale(img2)

        #Call Sift class
        sift = Sift()
        kpAS, descAS, kpBS, descBS, good_matches_sift = sift.match(tr_grey, test_grey)
        #Call Akaze class
        akaze = AKaze()
        kpAK, descAK, kpBK, descBK, good_matches_akaze = akaze.match(tr_grey, test_grey)
        #Call ORB class
        orb = ORB()
        kpAorb, descAorb, kpBorb, descBorb, good_matches_orb = orb.match(tr_grey, test_grey)

        #Mask for all three algorithm
        maskS, dtsS = location_extraction(kpAS, kpBS, good_matches_sift,tr_img)
        maskK, dtsK = location_extraction(kpAK, kpBK, good_matches_akaze,tr_img)
        maskorb, dtsorb = location_extraction(kpAorb, kpBorb, good_matches_orb,tr_img)


        #Copy of image so that lines from first method do not last to second.
        tmp_img1 = copy.copy(test_img)
        tmp_img2 = copy.copy(test_img)
        res_sift_img = create_results(tr_img, test_img, kpAS, kpBS, dtsS,
                                            maskS, good_matches_sift)
        res_akaze_img = create_results(tr_img, tmp_img1, kpAK, kpBK, dtsK,
                                            maskK, good_matches_akaze)

        res_orb_img = create_results(tr_img, tmp_img2, kpAorb, kpBorb, dtsorb,
                                            maskorb, good_matches_orb)

        #Create a table with the number of matches for each algorithm
        table_img = create_table(res_orb_img, good_matches_sift, good_matches_akaze, good_matches_orb)
        #Draw plots for the resulting images
        draw_plots(res_sift_img, res_akaze_img, res_orb_img, table_img, i, directory)

        i = i+1


#Read a folder containing images and return a list of urls
def read_folders(path):

    #List of named of all training images
    names = [f for f in os.listdir(path)]
    return names

def read_nr_images(path):
    images = [im for im in os.listdir(path) if os.path.isfile(os.path.join(path, im))]
    return len(images)

#Read an image and return its grey picture
def load__greyScale(picture):
    img = cv2.imread(picture)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, grey

#Takes keypoint descriptor and extracts its location
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

#Create a result image showing the detected matches if the algorithm
#successfully found the object. Otherwise it presents an image with
#all the detected keypoints
def create_results(tr_img, tmp_img, kpA, kpB, dst, matchesMask, good_matches):
    if dst != None:
        tmp_img = cv2.polylines(tmp_img,[np.int32(dst)],True, 255,3,cv2.LINE_AA)
        draw_params = dict(matchColor = (0, 255, 0),
                                singlePointColor = None,
                                matchesMask = matchesMask, # draw only inliers
                                flags = 2)
        res_img = cv2.drawMatches(tr_img,kpA,tmp_img,kpB,good_matches,None,**draw_params)

        return res_img
    else:
        #Draw keypoints that were detected for each picture
        res_img1 = cv2.drawKeypoints(tr_img, kpA, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        res_img2 = cv2.drawKeypoints(tmp_img, kpB, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #Get size of both pictures
        h1, w1 , d1 = res_img1.shape
        h2, w2, d2 = res_img2.shape

        #create an empty array with the size to hold both pictures
        res_img = np.zeros((max(h1,h2), w1+w2, d1), np.uint8)
        res_img[:h1, :w1] = res_img1
        res_img[:h2, w1:w1+w2] = res_img2

        return res_img

def create_table(res_orb_img, good_matches_sift, good_matches_akaze, good_matches_orb):
    h, w, d = res_orb_img.shape
    table_img = np.zeros((h,w,d), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    nr_matches_sift = 'Number of Sift matches: ' + str(len(good_matches_sift))
    cv2.putText(table_img, nr_matches_sift, (100, 100), font, 1, (255,255,255), 2, cv2.LINE_AA)

    nr_matches_akaze = 'Number of AKaze matches: ' + str(len(good_matches_akaze))
    cv2.putText(table_img, nr_matches_akaze, (100, 200), font, 1, (255,255,255), 2, cv2.LINE_AA)

    nr_matches_orb = 'Number of Orb matches: ' + str(len(good_matches_orb))
    cv2.putText(table_img, nr_matches_orb, (100, 300), font, 1, (255,255,255), 2, cv2.LINE_AA)

    return table_img

def draw_plots(sift, akaze, orb, table, index, directory):
    plot.subplot(221), plot.imshow(sift), plot.title('Sift')
    plot.subplot(222), plot.imshow(akaze), plot.title('AKaze')
    plot.subplot(223), plot.imshow(orb), plot.title('ORB')
    plot.subplot(224), plot.imshow(table), plot.title('Table')

    if not os.path.isdir(directory):
        os.makedirs(directory)
    number = index
    fig_name = directory + '/res' + str(number) + '.png'
    plot.savefig(fig_name, format ='png', dpi = 600)

    #plot.show()


main()
