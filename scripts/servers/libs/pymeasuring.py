#!/usr/bin/env python
import rospy
import math
import os
import cv2
import time
import imutils
import numpy as np
from scipy.spatial import distance as dist
from utils import *

from cvision.msg import Object

import geometry as g

DISSIMILARITY_THRESHOLD = 0.8

FOVS = (43.78, 54.35, 65.47)    # vertical, horizontal, diagonal angles. Logitech 920FullHD (640, 480)

MM_TO_M = 0.001
AREA_MIN = 2000
AREA_MAX = 35000

BALK20 = 'F20_20_'
BALK40 = 'S40_40_'
BOLT = 'M20_100'
M = 'M'
R20 = 'R20'
BEARING ='Bearing'
MOTOR = 'Motor'

DESIRED_CONTOURE_NAMES = [BALK20, BALK40, BOLT, M, R20, BEARING, MOTOR]
# DESIRED_CONTOURE_NAMES = ['1']

CONTOUR_FILES_EXT = '.npz'


class Measuring:
    """ class contains functions for processing image """

    def __init__(self, imageInfo, length):
        self.length = length  # lazy stick

        # image size
        x, y, _ = imageInfo['shape']
        self.xy0 = (x, y)

        # ratio [mm/px]
        self.ratioHFOV = imageInfo['ratio'][0]
        self.ratioVFOV = imageInfo['ratio'][1]
        self.ratioDFOV = imageInfo['ratio'][2]  # 0.3058082527

        # center RF
        self.CRF = (y / 2, x / 2)
        self.imageRF = ((0, -self.CRF[1]), (self.CRF[0], 0))
        self.standardContours = []

        # reading predetermined contours
        dirPath = os.path.dirname(os.path.realpath(__file__))
        for fileName in DESIRED_CONTOURE_NAMES:
            fileName = dirPath + '/contours/' + fileName + CONTOUR_FILES_EXT
            with np.load(fileName) as X:
                cnt = [X[i] for i in X]
                self.standardContours.append(cnt)
        rospy.loginfo('CONTOURS WAS LOADED!')

    def orderPoints(self, pts):
        """ sort corners in CW order"""
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        return np.array([tl, tr, br, bl], dtype="float32")

    def getObject(self, contour, image=None, shape='undefined'):
        """
            - measuring coordinates of contour;
            - measuring orientation;
            - measuring dimensions;
            - draw any info on frame
        """
        rospy.loginfo('OBJECT is ' + shape + '. Measuring...')
        obj = Object()
        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else -cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # box = perspective.order_points(box)
        box = self.orderPoints(box)
        if image is not None:
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 1)
        # coordinates of corners
        (tl, tr, br, bl) = box
        "object's position"
        objGRF = g.midpoint(tl, br)
        objCRF = (self.CRF[0] - objGRF[0], self.CRF[1] - objGRF[1])
        "object's dimensions [mm]"
        # middles of sides
        (tltrX, tltrY) = g.midpoint(tl, tr)
        (blbrX, blbrY) = g.midpoint(bl, br)
        (tlblX, tlblY) = g.midpoint(tl, bl)
        (trbrX, trbrY) = g.midpoint(tr, br)
        # compute the Euclidean distance between the midpoints of sides
        dX = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dY = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        dD = math.sqrt(dX ** 2 + dY ** 2)
        # for diagonal FOV
        dimD = dD * self.ratioDFOV

        # TODO SETUP for small and big MSize
        kMSize = 0.035
        if shape == M:
            if dimD < kMSize:
                shape = shape + '20'
            else:
                shape = shape + '30'

        alpha = math.atan2(dY, dX)
        dimX = dimD * math.cos(alpha)
        dimY = dimD * math.sin(alpha)
        "object's orientation. [-90; 90] from the X vector in the image RF"
        if dX >= dY:
            objVectorX = (tltrX - blbrX, tltrY - blbrY)
            angle = -g.angleBetween(self.imageRF[0], objVectorX)
        else:
            objVectorY = (tlblX - trbrX, tlblY - trbrY)
            angle = -g.angleBetween(self.imageRF[0], objVectorY)
        if abs(angle) > math.pi / 2:
            # if angle must be positive but it is not!!
            if 0 < abs(angle) < math.pi / 4 and tl[0] < br[0]:
                angle += -math.pi    
            else:
                angle += math.pi

        # for circle objects
        M20 = M + '20'
        M30 = M + '30'
        if shape in [BEARING, M20, M30]:
            angle = 0

        "!!!just fine form"
        dimPx = (dX, dY, dD)
        dimMm = (dimX, dimY, 0)
        dimM = (dimX, dimY, 0)
        objCRFinM = (objCRF[1] * self.ratioDFOV,
                     objCRF[0] * self.ratioDFOV,
                     -self.length)  # holy cow!
        objOrientation = (0, -angle, 0)

        rospy.loginfo('dims of object in [px]: ' + str(dimPx))

        "sent to ALEX server, but now it is not need, or noo, need, or no, not need again"
        # if self.flag:
        #     self.flag = False
        #     self.sendObject(objOrientation, objCRFinM)
        "packing object"
        obj.shape = shape
        obj.dimensions = dimM
        obj.coordinates_center_frame = objCRFinM
        obj.orientation = objOrientation
        for (x, y) in box:
            # top left corner of a object
            cv2.circle(image, (int(tl[0]), int(tl[1])), 5, (255, 0, 0), 2)

            cv2.putText(image, "{0:.1f}".format(angle * 180 / math.pi),
                        (int(0 + 50), int(0 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # BLUE point of a center object
            cv2.circle(image, (int(objGRF[0]), int(objGRF[1])), 5, (255, 0, 0), 2)
            cv2.putText(image, shape,
                        (int(objGRF[0]), int(objGRF[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # cross in a center object
            # dA
            cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 1)
            # dB
            cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 1)

            cv2.putText(image, 'A',
                        (int(tltrX), int(tltrY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, 'B',
                        (int(tlblX), int(tlblY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return obj, image

    def getListObjects(self, image, debug=False):
        """
            - filter frame;
            - find contours;
            - filter contours;
            - detect shape;
            - collect it all together
        :param image: raw image form camera
        :return:
            - list of collect' objects
            - painted image
            - state = Thue, if objects was found
        """
        listObjects = []

        # filtering image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if debug:
            bl = cv2.getTrackbarPos('bl', 'f') * 2 + 1

            t = cv2.getTrackbarPos('th', 'f')
            k = cv2.getTrackbarPos('k', 'f')
            c = cv2.getTrackbarPos('c', 'f')

            d = cv2.getTrackbarPos('d', 'f')
            e = cv2.getTrackbarPos('e', 'f')

            # bl, t, k, c = 2*2+1, 255, 14, 12
            blur = cv2.medianBlur(gray, 9)  # 3 work
            _, th = cv2.threshold(blur, 100, 255, cv2.THRESH_OTSU)
            # th = cv2.adaptiveThreshold(blur, t, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2*k+1, c)

            # v = np.median(blur)
            # sigma = 0.33
            # canny_low = int(max(0, (1 - sigma) * v))
            # canny_high = int(min(255, (1 + sigma) * v))
            #
            # th = cv2.Canny(blur, canny_low, canny_high)
            # edged = cv2.dilate(th, None, iterations=d)
            # th = cv2.erode(edged, None, iterations=e)

            cv2.imshow('blur', blur)
            cv2.imshow('thresh', th)
        else:
            # blur = cv2.medianBlur(gray, 2*2+1)
            # th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15*2+1, 13)
            blur = cv2.medianBlur(gray, 9)  # 3 work
            _, th = cv2.threshold(blur, 180, 255, cv2.THRESH_OTSU)

            # v = np.median(th)
            # sigma = 0.33
            # canny_low = int(max(0, (1 - sigma) * v))
            # canny_high = int(min(255, (1 + sigma) * v))
            #
            # th = cv2.Canny(th, canny_low, canny_high)
            # edged = cv2.dilate(th, None, iterations=7)
            # th = cv2.erode(edged, None, iterations=3)

        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # draw all contours
        cv2.drawContours(image, contours, -1, (0,0,255))

        rospy.loginfo('total contours in the frame is ' + str(len(contours)))

        # remove all the smallest and the biggest contours
        # just both arrays clearing from trash
        wellContours = []
        wellHierarchy = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if AREA_MIN < area < AREA_MAX:
                wellContours.append(contour)
                wellHierarchy.append(hierarchy[0][i])

        if debug:
            cv2.drawContours(image, wellContours, -1, (0, 0, 255))

        # compare and selecting desired contours
        foundContours = []
        for i, wCnt in enumerate(wellContours):
            obj = None
            okCntsForOne = []
            h = wellHierarchy[i]
            for j, sCnt in enumerate(self.standardContours):
                shape = DESIRED_CONTOURE_NAMES[j]
                if True:    # h[3] in [-1,1,2,3,4,5]:
                    ret = cv2.matchShapes(sCnt[0], wCnt, 1, 0)
                    if obj is None or ret < obj[0]:
                        obj = (ret, wCnt, shape)
                        okCntsForOne.append(obj)
            minimum = 999
            okObj = None
            for k in okCntsForOne:
                if k[0] < minimum:
                    minimum = k[0]
                    okObj = k
            foundContours.append(okObj)

        # for i, sCnt in enumerate(self.standardContours):
        #     obj = None
        #     shape = DESIRED_CONTOURE_NAMES[i]
        #     for i, wCn0t in enumerate(wellContours):
        #         # p = cv2.arcLength(wCnt, True)
        #         # wCnt = cv2.approxPolyDP(wCnt, 0.005 * p, True)
        #         h = wellHierarchy[i]
        #         if h[3] == -1:
        #             ret = cv2.matchShapes(wCnt, sCnt[0], 1, 0)
        #             if obj is None or ret < obj[0]:
        #                 obj = (ret, wCnt, shape)
        #                 foundContours.append(obj)

        test = []
        try:
            for f in foundContours:
                test.append(f[1])

            # draw OK conours
            cv2.drawContours(image, test, -1, (255, 0, 0))

            rospy.loginfo('qty well contours: ' + str(len(wellContours)))
            rospy.loginfo('qty selected contours by matching: ' + str(len(foundContours)))
            print('*******************\n')

            # matching and measuring found contours
            if len(foundContours) != 0:
                for cnt in foundContours:
                    if cnt[0] < DISSIMILARITY_THRESHOLD:
                        print(cnt[0], cnt[2])
                        rospy.loginfo(cnt[2].upper() +
                                      ' | area: ' + str(cv2.contourArea(cnt[1])) +
                                      ' similr.: ' + str(cnt[0]))

                        mask = np.zeros(gray.shape[:2], dtype='uint8')
                        cv2.drawContours(mask, [cnt[1]], -1, 255, -1)

                        if cnt[2] in [BALK20, BALK40]:
                            mean = cv2.mean(gray, mask)
                            if debug:
                                cv2.imshow('ssss', mask)

                            # !!! TODO SETUP for black and wihite balks
                            kMean = 70
                            if mean[0] < kMean:
                                o, image = self.getObject(cnt[1], image, shape=cnt[2]+'B')
                            else:
                                o, image = self.getObject(cnt[1], image, shape=cnt[2]+'G')
                        else:
                            o, image = self.getObject(cnt[1], image, shape=cnt[2])
                        # print(cnt[2])
                        listObjects.append(o)

        except TypeError:
            pass

        # RED point in a center frame
        cv2.circle(image, (int(self.CRF[0]), int(self.CRF[1])), 5, (0, 0, 255), 2)
        # cross in a center frame
        cv2.line(image, (0, self.CRF[1]), (self.xy0[1], self.CRF[1]), (0, 0, 255), 2)
        cv2.line(image, (self.CRF[0], 0), (self.CRF[0], self.xy0[0]), (0, 0, 255), 1)

        # scale for convenient
        scale = 1
        image = cv2.resize(image, (int(scale * self.xy0[1]), int(scale * self.xy0[0])))

        # TODO think about it :)
        state = True
        if len(listObjects) > 0:
            state = False
        return listObjects, image, state


if __name__ == '__main__':


    cv2.namedWindow('f')

    def nothing(x):
        pass

    cv2.createTrackbar('th', 'f', 255, 255, nothing)
    cv2.createTrackbar('k', 'f', 14, 21, nothing)
    cv2.createTrackbar('c', 'f', 12, 21, nothing)
    cv2.createTrackbar('bl', 'f', 2, 21, nothing)

    cv2.createTrackbar('d', 'f', 9, 21, nothing)
    cv2.createTrackbar('e', 'f', 6, 21, nothing)


    l = 0.565

    # cap = cv2.VideoCapture(1)
    # _, frame = cap.read()

    file = 'draft/photo.jpg'
    frame = cv2.imread(file)
    shape = frame.shape

    imageInfo = dict()
    imageInfo['shape'] = (shape[0], shape[1], math.sqrt(shape[0] ** 2 + shape[1] ** 2))
    imageInfo['ratio'] = None  # ration [mm] / [px]

    imageDim = getDimImage(l, 0, 0, 78)  # 54.5, 42.3, 66.17
    imageInfo['ratio'] = getRatio(imageInfo['shape'], imageDim)

    m = Measuring(imageInfo, l)
    SCALE = 0.7

    while True:
        frame = cv2.imread(file)
        w, h, _ = frame.shape
        frame = cv2.resize(frame, (int(h * SCALE), int(w * SCALE)))
        # _, frame = cap.read()
        # frame[340:480, 1:640] = [0,0,0]
        w, h, _ = frame.shape
        frame = cv2.resize(frame, (int(h * SCALE), int(w * SCALE)))
        #
        list, image, state = m.getListObjects(frame.copy(), debug=True)
        image = cv2.resize(image, (int(h * SCALE), int(w * SCALE)))

        w, h, _ = image.shape
        image = cv2.resize(image, (int(h * 0.5), int(w * 0.5)))

        cv2.imshow('0', image)
        if not state:
            print(list)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(1)

    # cap.release()
    cv2.destroyAllWindows()

