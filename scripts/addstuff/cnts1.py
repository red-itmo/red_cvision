import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Bars')

cv2.createTrackbar('sigma', 'Bars', 33, 100, nothing)
cv2.createTrackbar('blur', 'Bars', 1, 15, nothing)
cv2.createTrackbar('threshold kernel', 'Bars', 9, 20, nothing)
cv2.createTrackbar('c', 'Bars', 5, 10, nothing)
cv2.createTrackbar('dilate', 'Bars', 5, 10, nothing)
cv2.createTrackbar('erode', 'Bars', 1, 10, nothing)

if __name__ == '__main__':
    # VideoCapture(1) FOR LOGITECH
    cam = cv2.VideoCapture(0)
    image = None
    while True:
        _, frame = cam.read()
        # image = frame[0:360, 0:640]
        image = frame

        # HSV
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # # V
        # hsv[:,:,2] = 0
        # # S
        # hsv[:,:,1] = 0
        # # H
        # hsv[:,:,0] = hsv[:,:,0]
        # # hsv[:,:][:,:,:] = hsv[:,:][:,0,:]
        # # print(hsv[0,0])

        w, h, _ = image.shape

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = 1 + 2 * cv2.getTrackbarPos('blur', 'Bars')
        # gray = cv2.blur(gray, (blur, blur))
        gray = cv2.medianBlur(gray, blur)
        # image = cv2.blur(image, (7, 7))
        # gray = cv2.bitwise_not(image)


        #THRESHOLDING
        thkernel = 3 + 2 * cv2.getTrackbarPos('threshold kernel', 'Bars')
        thparam = cv2.getTrackbarPos('c', 'Bars')
        # th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,thkernel,thparam)
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,thkernel,thparam)

        dilate_iter = cv2.getTrackbarPos('dilate', 'Bars')
        erode_iter = cv2.getTrackbarPos('erde', 'Bars')

        #CANNY
        v = np.median(gray)
        sigma = cv2.getTrackbarPos('sigma', 'Bars')/100
        # sigma = 0.33
        dilate_iter = cv2.getTrackbarPos('dilate', 'Bars')
        erode_iter = cv2.getTrackbarPos('erode', 'Bars')
        canny_low = int(max(0, (1 - sigma) * v))
        canny_high = int(min(255, (1 + sigma) * v))
        edged = cv2.Canny(th, canny_low, canny_high)

        dilate_iter = cv2.getTrackbarPos('dilate', 'Bars')
        erode_iter = cv2.getTrackbarPos('erde', 'Bars')
        edged = cv2.dilate(edged, None, iterations=dilate_iter)
        edged = cv2.erode(edged, None, iterations=3)

        _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gray, contours, -1, (0, 255, 0))
        
        # print(len(contours))
        p = cv2.arcLength(contours[0], True)
        c = cv2.approxPolyDP(contours[0], 0.001 * p, True)
        
        cv2.drawContours(image, c, -1, (0, 0, 255), 3)
        
        # np.savez('squar50', c)
        
        # with np.load('squar50.npz') as X:
        #     c = [X[i] for i in X]
        # cv2.drawContours(image, c, -1, (0, 0, 255), 3)


        cv2.imshow('lol1', th)
        cv2.imshow('lol', gray)
        # cv2.imshow('image_test', image_test)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

