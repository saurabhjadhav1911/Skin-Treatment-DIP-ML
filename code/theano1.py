#C:\Users\Public\Skin Treatment DIP_ML\code
from theano import *
from theano import tensor as T
import numpy as np
import cv2
import ImageLoader
rad=np.random

row=0
col=0
IL=ImageLoader.Neural_Image_Loader()
def get_Image():
	global row,col
	img,mask=IL.NextSet()
	row,col,chann=img.shape
	x=img.reshape((row*col),chann)
	y=mask.reshape((row*col))
	yp1=y>128
	yp2=y<129
	yp1=yp1*1
	yp2=yp2*1
	yp=np.array([yp1,yp2])
	yp=yp.transpose()
	return x,yp
Xi,Yi=get_Image()
print(Xi,Yi)
rng = np.random.RandomState(1234)
#network sizes
training_steps=1000

#Data input vaiable

D=(np.asarray(Xi),np.asarray(Yi))

#data variables
def layer(n_in, n_out):
    return theano.shared(value=np.asarray(rng.uniform(low=-1.0, high=1.0, size=(n_in, n_out)), dtype=theano.config.floatX), name='W', borrow=True)

X = theano.shared(np.asarray(Xi), name='X')
y = theano.shared(np.asarray(Yi), name='y')

#layer weights
w_1=layer(3,5)
w_2=layer(5,1)

print (w_1.get_value())
print (w_2.get_value())

#expression
y_1=T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(X,w_1)),w_2))
prediction = y_1 > 0.5

#cross entropy 
xent=-y*T.log(y_1)-(1-y)*T.log(1-y_1)

#cost function

cost=T.sum((y-y_1)**2)/2
LEARNING_RATE=0.01
#gradients
updates = [(w_1, w_1 - LEARNING_RATE * T.grad(cost, w_1)), (w_2, w_2 - LEARNING_RATE * T.grad(cost, w_2))]

#compile
train = theano.function(inputs = [],   outputs = [],  updates =updates)
predict = theano.function(inputs = [], outputs = y_1)

res=np.array(predict())
res=255*res
res=res.reshape((row,col))
print(res)
cv2.imshow('skin_neural',res)
cv2.waitKey(0)
# Train 
for i in range(training_steps):
	train()  
#print("Final model")
#print(w_1.get_value(), b_1.get_value())
#print(w_2.get_value(), b_2.get_value())
res=np.array(predict())
print(res)
res=255*res
res=res.reshape((row,col))
print(res)
res=res.astype(np.uint8)
print(res)
cv2.imshow('skin_neural',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
