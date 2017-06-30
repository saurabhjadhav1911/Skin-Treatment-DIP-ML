#C:\Users\Public\Skin Treatment DIP_ML\code
import cv2
import os.path
import numpy as np
from os import listdir
from itertools import cycle
from os.path import isfile, join

DataBase_Path=join(os.path.split(os.getcwd())[0],"Image_Data\Acne")#"C:/Users/Public/Skin Treatment DIP_ML/Image_Data/Ringworm"
DataBase_Path_X=join(os.path.split(os.getcwd())[0],"Image_Data\ORI")#"
DataBase_Path_Y=join(os.path.split(os.getcwd())[0],"Image_Data\GT")#"
#this class reads all sequenced images from database folder
class Image_Loader():
	def __init__(self):
		self.Dict={}
		self.alphabets='abcdefghijklmnopqrstuvwxyz'
		self.Image_files = [f for f in listdir(DataBase_Path) if isfile(join(DataBase_Path, f))]
		for f in self.Image_files:
			if ('.jpg' in (f.lower())) and (not(f[f.index('.')-1].isdigit())):
				n_array=[int(s) for n,s in enumerate(f) if s.isdigit()]
				Image_Number=0
				for n,i in enumerate(n_array):
					Image_Number+=i*(10**(len(n_array)-n-1))
				try:
					self.Dict[str(Image_Number)].append(f)
				except:
					self.Dict.update({str(Image_Number):[f]})
		self.Iter=cycle(self.Dict)
	def NextSet(self):
		set_num=next(self.Iter, '-1')
		set_names=self.Dict[set_num]
		set_names.sort()
		img=[cv2.imread(join(DataBase_Path, f)) for f in set_names if f]
		return img
	def NextBatch(self,size):
		batch=np.array([self.NextSet() for _ in xrange(size)],dtype=object)
		return batch

class Neural_Image_Loader():
	def __init__(self):
		self.Dict={}
		self.row=0
		self.col=0
		self.Image_filesX = [f for f in listdir(DataBase_Path_X) if isfile(join(DataBase_Path_X, f))]
		self.Image_filesY = [f for f in listdir(DataBase_Path_Y) if isfile(join(DataBase_Path_Y, f))]
		for f in self.Image_filesX:
			if ('.jpg' in (f.lower())) and (not(f[f.index('.')-1].isdigit())):
				n_array=[int(s) for n,s in enumerate(f) if s.isdigit()]
				Image_Number=0
				for n,i in enumerate(n_array):
					Image_Number+=i*(10**(len(n_array)-n-1))
				self.Dict.update({str(Image_Number):f})
		self.Iter=cycle(self.Dict)
	def NextSet(self):
		set_num=next(self.Iter, '-1')
		f=self.Dict[set_num]
		print("f",f)
		img=cv2.imread(join(DataBase_Path_X, f))
		mask=cv2.imread(join(DataBase_Path_Y, f))
		mask=cv2.inRange(mask,(10,10,10),(255,255,255))
		return img,mask
	def NextBatch(self,size):
		batchx=[]
		batchy=[]
		for _ in xrange(size):
			x,y=self.NextSet()
			batchx.append(x)
			batchy.append(y)
		return batchx,batchy
	def get_Image_Batch(self):
		img,mask=self.NextSet()
		self.row,self.col,chann=img.shape
		x=img.reshape((self.row*self.col),chann)
		y=mask.reshape((self.row*self.col))
		yp1=y>128
		yp2=y<129
		yp1=yp1*1
		yp2=yp2*1
		yp=np.array([yp1,yp2])
		yp=yp.transpose()
		return x,yp
if __name__ == "__main__":

	IL=Neural_Image_Loader()
	X,Y=IL.get_Image_Batch()
	Y=Y.transpose()
	Y=Y[0]
	print(X,Y)
