import cv2

class ORB:

    def __init__(self):
        self.orb = cv2.ORB_create()

    def match(self, grey1, grey2):
        word = None
        kpA, descA = self.orb.detectAndCompute(grey1, word)
        kpB, descB = self.orb.detectAndCompute(grey2, word)

        return self.bf_matching(kpA, descA, kpB, descB)

    def bf_matching(self, kpA, descA, kpB, descB):
        bfm =  cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = bfm.match(descA, descB)

        good_matches = sorted(matches, key = lambda x:x.distance)

        return kpA, descA, kpB, descB, good_matches
