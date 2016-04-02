import cv2

#Class that calls for openCV function akaze
class AKaze:
    #Constructor for objects of class AKaze
    def __init__(self):
        self.akaze = cv2.AKAZE_create()

    #Function that detects keypoints in image and creates a
    #descriptor for each one
    def match(self, grey1, grey2):
        word = None

        #start tracking time
        e1 = cv2.getTickCount()
        kpA, descA = self.akaze.detectAndCompute(grey1, word)
        kpB, descB = self.akaze.detectAndCompute(grey2, word)

        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        return self.bf_matching(kpA, descA, kpB, descB, time)

    #Uses brute force matching
    def bf_matching(self, kpA, descA, kpB, descB, time):
        bfm = cv2.BFMatcher()
        matches = bfm.knnMatch(descA, descB, k = 2)

        #Ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        return kpA, descA, kpB, descB, good_matches, time
