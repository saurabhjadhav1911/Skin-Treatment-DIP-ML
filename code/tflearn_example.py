import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X,Y,test_X,test_Y=mnist.load_data(one_hot=True)

conv_sizes=[32,64]
NN_sizes=[1024,10]
activation_fn=['relu','softmax']
X=X.reshape([-1,28,28,1])
test_X=test_X.reshape([-1,28,28,1])

y=input_data(shape=[None,28,28,1],name='input')

for n in conv_sizes:
	y=conv_2d(y,n,2,activation='relu')
	y=max_pool_2d(y,2)

y=fully_connected(y,NN_sizes[0],activation='relu')
y=dropout(y,0.8)

y=fully_connected(y,NN_sizes[1],activation='relu')
y=regression(y,optimizer='adam',learning_rate=0.01,loss='categorical_crossentropy',name='target')

model=tflearn.DNN(y)

model.fit({'input':X},{'target':Y},n_epoch=10,
	validation_set=({'input':test_X},{'target':test_Y}),
	snapshot_step=500,show_metric=True,run_id='mnist')

model.save('tflearn_mnist_example')