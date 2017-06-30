#C:\Users\Public\Skin Treatment DIP_ML\code
import cv2
import ImageLoader
import numpy as np
from matplotlib import pyplot as plt
import argparse
u=0.61549
s=0.0203031

def Hist(img):
	color = ('b','g','r')
	for i,col in enumerate(color):
	    histr = cv2.calcHist([img],[i],None,[256],[0,256])
	    plt.plot(histr,color = col)
	    plt.xlim([0,256])
	plt.show()
	cv2.waitKey(0)
def f(x):
	global u,s
	e=np.exp((-1*(x-u)/s))
	F=e/(s*((1+e)**2))
	return F
f = np.vectorize(f, otypes=[np.float])
def Skin(img):
	
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	H,S,V=cv2.split(hsv)
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	height, width = img.shape[:2]
	img = cv2.resize(img,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
	blur = cv2.GaussianBlur(img,(3,3),0)
	skin=cv2.adaptiveThreshold(hsv[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	cv2.imwrite('hdg.jpg',blur)
	#Hist(hsv)
	F=f(H)
	fac=255/np.max(F)
	F=F*fac
	F.astype(int)
	print F
	cv2.imshow("skin",blur)
	cv2.waitKey(0)

IL=ImageLoader.Image_Loader()
images=IL.NextSet()
for image in images:
	#res=FeatureDetect(image)
	cv2.imshow("ip",image)
	Skin(image)
	#cv2.imshow("res",res)
	cv2.waitKey(0)
cv2.destroyAllWindows()