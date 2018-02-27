import cv2
import numpy as np


if __name__ == '__main__':

    # PATH TO THE IMAGE
    image_path = "scew_small.png"
    image = cv2.imread(image_path, 1)
    
    # SCALING 
    scale = 1
    # image = cv2.imread('circle.png')
    w, h, _ = image.shape
    image = cv2.resize(image, (int(h*scale), int(w*scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # BLURING
    blur = 1
    # gray = cv2.blur(gray, (blur, blur))
    gray = cv2.medianBlur(gray, blur)
    # image = cv2.blur(image, (7, 7))
    # gray = cv2.bitwise_not(image)


    #THRESHOLDING
    thkernel = 9
    thparam = 4
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,thkernel,thparam)


    #CANNY
    v = np.median(gray)
    sigma = cv2.getTrackbarPos('sigma', 'Bars')/100
    # sigma = 0.33
    canny_low = int(max(0, (1 - sigma) * v))
    canny_high = int(min(255, (1 + sigma) * v))
    edged = cv2.Canny(th, canny_low, canny_high)
    
    # OPENING
    dilate_iter = 5
    erode_iter = 1
    edged = cv2.dilate(edged, None, iterations=dilate_iter)
    edged = cv2.erode(edged, None, iterations=erode_iter)

    _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0))
    
    print(len(contours))
    p = cv2.arcLength(contours[0], True)
    c = cv2.approxPolyDP(contours[0], 0.01 * p, True)
    
    cv2.drawContours(image, c, -1, (0, 0, 255), 3)
    
    np.savez('scew_small', c)
    
    # ??? 
    # with np.load('squar50.npz') as X:
    #     c = [X[i] for i in X]
    # cv2.drawContours(image, c, -1, (0, 0, 255), 3)
    # PROFIT

    cv2.imshow('lol1', th)
    cv2.imshow('lol', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
