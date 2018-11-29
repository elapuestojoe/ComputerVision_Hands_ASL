import numpy as np
import cv2
import math

#LOADING HAND CASCADE
# hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')
# hand_cascade = cv2.CascadeClassifier('hand_cascade_2.xml')
hand_cascade = cv2.CascadeClassifier('Hand.Cascade.1.xml')

# VIDEO CAPTURE
cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 20)
while 1:
	ret, img = cap.read()
	blur = cv2.GaussianBlur(img,(5,5),0) # BLURRING IMAGE TO SMOOTHEN EDGES
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) # BGR -> GRAY CONVERSION
	retval1 ,thresh1 = cv2.threshold(gray,75,255,cv2.THRESH_OTSU) # THRESHOLDING IMAGE
	# retval1 ,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.ADAPTIVE_THRESH_GAUSSIAN_C) # THRESHOLDING IMAGE
	# retval1 ,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.ADAPTIVE_THRESH_MEAN_C ) # THRESHOLDING IMAGE

	hands = hand_cascade.detectMultiScale(thresh1, 1.3, 5) # DETECTING HAND IN THE THRESHOLDE IMAGE

	mask = np.zeros(thresh1.shape, dtype = "uint8") # CREATING MASK

	print(len(hands))

	for (x,y,w,h) in hands: # MARKING THE DETECTED ROI
		cv2.rectangle(img,(x,y),(x+w,y+h), (122,122,0), 2) 
		cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)

	img2 = cv2.bitwise_and(thresh1, mask)

	cv2.imshow("Original", img)
	cv2.imshow("Detected", thresh1)
	cv2.imshow("IMG2", img2)

	# final = cv2.GaussianBlur(img2,(7,7),0)	
	# contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	k = cv2.waitKey(30)
	if(k == 27):
		break