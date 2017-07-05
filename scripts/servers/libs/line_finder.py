#!/usr/bin/env python
import cv2
import time


class LineFinder:

    def __init__(self):
        pass

    def getLine(self, image):
        cv2.putText(image, "LINE DON'T FOUND!\ntime is " + str(time.time()), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
        return image
