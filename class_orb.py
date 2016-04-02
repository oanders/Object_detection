import cv2

class ORB:

    def __init__(self):
        self.orb = cv2.ORB_create()

    def match(self, grey1, grey2):
        word = None

        #start tracking time
        e1 = cv2.getTickCount()
        kpA, descA = self.orb.detectAndCompute(grey1, word)
        kpB, descB = self.orb.detectAndCompute(grey2, word)

        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        return self.bf_matching(kpA, descA, kpB, descB, time)

    def bf_matching(self, kpA, descA, kpB, descB, time):
        bfm =  cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = bfm.match(descA, descB)

        good_matches = sorted(matches, key = lambda x:x.distance)

        return kpA, descA, kpB, descB, good_matches, time
