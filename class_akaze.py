import cv2

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
