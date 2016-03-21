import cv2

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
