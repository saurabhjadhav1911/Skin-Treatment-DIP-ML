#C:\Users\Public\Skin Treatment DIP_ML\code
import cv2
import ImageLoader
import numpy as np
from matplotlib import pyplot as plt
import argparse


def FeatureDetect(img):
	#1.Convert the RGB color images to the Grey scaleimages
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#2. Find the maximum value of intensity images with X and Y coordinates on the Grey scale images.
	#3. Calculate normalized grey-scale image by divide the value of intensity to 0 or 1 with X and Y coordinates, to compare with HSV images.
	norm_gray=Normalise(gray)
	#4. Retrieve HSV color images to define the value of H(Hue) = 0 for drop a red color
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	H=hsv[:,:,0]
	Red=cv2.inRange(H,0,6)
	cv2.imshow("red",Red)
	#5. To extract the brightness area (V) from HSV model  and define Dark color = 0 and White color = 1.
	V=hsv[:,:,2]

	norm_V=Normalise(V)
	#6. To subtract by V-Grey scale, the result show the region of maximum lightness
	v_gr=norm_V-norm_gray
	norm_gray=norm_gray
	cv2.imshow("gray",norm_gray)
	cv2.imshow("v",norm_V)
	#7. Define the value of threshold background is white color otherwise will be a black color. The images convert to negative binary color
	#8. To analyse the images for eliminate a tiny spot area.
	#9. From the result of step 8, divided the area less than7000.
	#10. The results from step 7, 8 and 9 will represent the
	v_gr=v_gr
	#print(v_gr)
	return v_gr

def GLCM(img):
	mat=np.zeros((255,255,3),dtype=np.uint32)
	row,col=img.shape
	for r in xrange(1,row):
		for c in xrange(1,col):
			mat[img[r-1][c]][img[r][c]][0]+=1
			mat[img[r][c-1]][img[r][c]][1]+=1
			mat[img[r-1][c-1]][img[r][c]][2]+=1

	return mat
def Normalise(mat):
	print(np.max(mat))
	sum=1.0/np.max(mat)
	print('faCTOR:',sum)
	mat=mat*sum
	return mat
def energy(mat):
	enn=0
	mat=Normalise(mat)
	for row in mat:
		for col in row:
			enn+=col**2
	return enn
	
IL=ImageLoader.Image_Loader()
images=IL.NextSet()
images=IL.NextSet()
'''red=np.ze ros((500,500,3),dtype=np.uint8)
hsv = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
hsv[:,:,2]=255
hsv[:,:,1]=255
hsv[:,:,0]=255
red = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
'''

for image in images:
	res=FeatureDetect(image)
	cv2.imshow("ip",image)
	cv2.imshow("res",res)
	cv2.waitKey(0)
cv2.destroyAllWindows()
'''
1. Convert the RGB color images to the Grey scale
images
2. Find the maximum value of intensity images with
X and Y coordinates on the Grey scale images.
3. Calculate normalized grey-scale image by divide
the value of intensity to 0 or 1 with X and Y
coordinates, to compare with HSV images.
4. Retrieve HSV color images to define the value of
H(Hue) = 0 for drop a red color
5. To extract the brightness area (V) from HSV model
and define Dark color = 0 and White color = 1.
6. To subtract by V-Grey scale, the result show the
region of maximum lightness
7. Define the value of threshold background is white
color otherwise will be a black color. The images
convert to negative binary color
8. To analyse the images for eliminate a tiny spot
area.
9. From the result of step 8, divided the area less than
7000.
10. The results from step 7, 8 and 9 will represent the
'''