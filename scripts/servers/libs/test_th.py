import cv2
import numpy as np
import time


SCALE = 1

def getFiltred2(src, blur, sigma, thkernel, thparam, dilate_iter, erode_iter):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, blur)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thkernel, thparam)

    v = np.median(th)
    sigma = sigma
    canny_low = int(max(0, (1 - sigma) * v))
    canny_high = int(min(255, (1 + sigma) * v))

    edged = cv2.Canny(th, canny_low, canny_high)
    th = cv2.dilate(edged, None, iterations=dilate_iter)
    th = cv2.erode(th, None, iterations=erode_iter)

    contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src, contours, -1, (0, 255, 0))
    c = []
    if len(contours) > 0:
        p = cv2.arcLength(contours[0], True)
        c = cv2.approxPolyDP(contours[0], 0.001 * p, True)
        cv2.drawContours(src, c, -1, (0, 0, 255), 3)

    return src, th, edged, c

def getFiltred(src, blur, sigma, thkernel, thparam, dilate_iter, erode_iter):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, blur)

    # th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,thkernel,thparam)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thkernel, thparam)
    # CANNY
    v = np.median(gray)

    # sigma = 0.33
    canny_low = int(max(0, (1 - sigma) * v))
    canny_high = int(min(255, (1 + sigma) * v))
    edged = cv2.Canny(th, canny_low, canny_high)

    edged = cv2.dilate(edged, None, iterations=dilate_iter)
    edged = cv2.erode(edged, None, iterations=erode_iter)

    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src, contours, -1, (0, 255, 0))

    if len(contours) > 0:
        p = cv2.arcLength(contours[0], True)
        c = cv2.approxPolyDP(contours[0], 0.001 * p, True)
        cv2.drawContours(src, c, -1, (0, 0, 255), 3)

    return src, th, edged, c


def nothing(data):
    pass

def main():

    cv2.namedWindow('Bars')

    cv2.createTrackbar('sigma', 'Bars', 33, 100, nothing)
    cv2.createTrackbar('blur', 'Bars', 1, 15, nothing)
    cv2.createTrackbar('threshold kernel', 'Bars', 9, 20, nothing)
    cv2.createTrackbar('c', 'Bars', 5, 10, nothing)
    cv2.createTrackbar('dilate', 'Bars', 5, 10, nothing)
    cv2.createTrackbar('erode', 'Bars', 3, 10, nothing)

    path = 'draft/'
    # images =  ['F20_20_B', 'F20_20_G', 'S40_40_B', 'S40_40_G', 'M20_100', 'M20','M30', 'R20',
    #             'Bearing_Box', 'Bearing', 'Axis', 'Distance_Tube', 'Motor']

    images = ['Motor']
    ext = '.png'
    # ext = '.jpg'

    pathDst = 'contours/'
    cap = cv2.VideoCapture(1)
    while True:
        _, image = cap.read()
        w, h, _ = image.shape
        image = cv2.resize(image, (int(h * SCALE), int(w * SCALE)))

        sigma = cv2.getTrackbarPos('sigma', 'Bars') / 100
        blur = 1 + 2 * cv2.getTrackbarPos('blur', 'Bars')
        thkernel = 3 + 2 * cv2.getTrackbarPos('threshold kernel', 'Bars')
        thparam = cv2.getTrackbarPos('c', 'Bars')
        dilate_iter = cv2.getTrackbarPos('dilate', 'Bars')
        erode_iter = cv2.getTrackbarPos('erode', 'Bars')

        gray, th, edged, cnt = getFiltred(image.copy(), blur, sigma, thkernel, thparam, dilate_iter, erode_iter)

        cv2.imshow('original', gray)
        cv2.imshow('thresh', th)
        cv2.imshow('edged', edged)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
