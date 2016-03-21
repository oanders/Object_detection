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
        kpA, descA = self.akaze.detectAndCompute(grey1, word)
        kpB, descB = self.akaze.detectAndCompute(grey2, word)

        return self.bf_matching(kpA, descA, kpB, descB)

    #Uses brute force matching
    def bf_matching(self, kpA, descA, kpB, descB):
        bfm = cv2.BFMatcher()
        matches = bfm.knnMatch(descA, descB, k = 2)

        #Ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        return kpA, descA, kpB, descB, good_matches
