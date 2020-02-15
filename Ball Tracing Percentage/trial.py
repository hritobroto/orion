#packages
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


#image preparation and conversion to HSV and creating mask using bitwise opereations
img = cv2.imread('D:/self12.jpg')
img = imutils.resize(img, width=600)
blurred = cv2.GaussianBlur(img, (11, 11), 0)
hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV);
low=np.array([25,52,72]);
high=np.array([102,255,255]);
mask=cv2.inRange(hsv,low,high);
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
green=cv2.bitwise_and(img,img,mask=mask);


contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contour = imutils.grab_contours(contour)
center = None
if len(contour) > 0:
    # finding the largest contour and use it to draw an enclosing circle around the outer surface of the ball
    c = max(contour, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    if radius > 10:
        # drawing outline over the ball
        cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
    
ratio_green = cv2.countNonZero(mask)/(img.size/3)
print('Percentage Green:', np.round(ratio_green*100, 2))
cv2.imshow("images", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
