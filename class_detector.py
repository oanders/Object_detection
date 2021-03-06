import cv2

#A class that calls for openCV function Sift
class Detector:
    #Contructor for class, objects of class SIFT
    def __init__(self, detector, argu):
        self.detector = detector
        self.argu = argu

    #Function that detects keypoints in image and creates a
    #descriptor for each one
    def match(self, grey1, grey2):
        word = None

        #start tracking time for detection
        e1 = cv2.getTickCount()
        kpA, descA = self.detector.detectAndCompute(grey1, word)
        kpB, descB = self.detector.detectAndCompute(grey2, word)

        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        return self.bf_matching(kpA, descA, kpB, descB, time)

    #Uses brute force matching
    def bf_matching(self, kpA, descA, kpB, descB, time):
        if self.argu == 'NORM_HAMMING':
            bfm =  cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            bfm = cv2.BFMatcher()
        matches = bfm.knnMatch(descA, descB, k = 2)

        #Ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        return kpA, descA, kpB, descB, good_matches, time
