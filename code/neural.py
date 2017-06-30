import tensorflow as tf
x1=tf.constant([[1,2,3],[4,5,6],[7,8,9]])
x2=tf.constant([[1,0,0],[0,1,0],[0,0,1]])


def sigmoid_prime(var):
    return(np.exp(-var)/((1+np.exp(-var))**2))

class NeralNetwiork(object):
	def __init__(self):
		self.LayerNeuronCount=tf.constant([2,3,1])
        self.LayerSize=tf.constant(len(self.LayerNeuronCount))
        #weights
        self.W=tf.array([tf.ones([self.LayerNeuronCount[layernum-1],self.LayerNeuronCount[layernum]],dtype=object) for layernum in range(1,self.LayerSize)])




with tf.Session() as sess:
	output=sess.run((result))
	print (output)

'''
import numpy as np

X=[[1,2],[3,4],[5,6],[7,8]]
Y=[7,8,9,10]
k=2

def sigmoid(var):
    return(1/(1+np.exp(-var)))

def sigmoid_prime(var):
    return(np.exp(-var)/((1+np.exp(-var))**2))

class Neural_Network(object):
    def __init__(self):
        #pass Hyperparamters
        self.LayerNeuronCount=[2,3,1]
        self.LayerSize=len(self.LayerNeuronCount)
        #weights
        self.W=np.array([np.ones([self.LayerNeuronCount[layernum-1],self.LayerNeuronCount[layernum]],dtype=object) for layernum in range(1,self.LayerSize)])
    def forward(self,x):
        #print self.A
        #self.A=np.append(self.A,[sigmoid(self.Z[ni])])
        for ni in range(1,self.LayerSize):
            if(ni==1):
                self.A=np.array([x],dtype=object)
                print self.A[ni-1]
                print self.W[ni-1]
                self.Z=np.array([np.dot(self.A[ni-1],self.W[ni-1])],dtype=float)

                print self.Z
            else:
                a=sigmoid(self.Z[ni-1])
                self.A=np.append(self.A,[a])
                z=np.array([np.dot(self.A[ni-1],self.W[ni-1])],dtype=float)
                print z
                self.Z=np.append(self.Z,[z],axis=0)
                print self.Z
            
            print "Z shape"
            print self.Z.shape
            print "A shape"
            print self.A.shape
            print "W shape"
            print self.W.shape
            
            print self.Z
            print self.A
            
        
        return self.A
    '''
    '''def costFunction(self,x,y):
        self.yhat=self.forward(x)
        J=0.5*np.sum(((y-self.yhat)**2))
        return J
    
    def costFunctionPrime(self,x,y):
        #compute cost
        self.yhat=self.forward(x)
        delta3=np.multiply((-y+self.yhat),sigmoid_prime(self.z3))
        dJdW2=np.dot(self.a2.T,delta3)
        delta2=np.dot(delta3,self.W2.T)*sigmoid_prime(self.z2)
        dJdW1=np.dot(x.T,delta2)
        return dJdW1,dJdW2
scalar=1'''
#NN=Neural_Network()

#print NN.forward(X)
'''cost1=NN.costFunction(X,Y)
print cost1
cost2=NN.costFunction(X,Y)
n=0
tolerance=400
while(abs(cost2)>tolerance):
    dJdW1,dJdW2=NN.costFunctionPrime(X,Y)
    NN.W1-=scalar*dJdW1
    NN.W2-=scalar*dJdW2
    cost2=NN.costFunction(X,Y)
    if((n%100)==0):
        print cost2
    n+=1
    #print cost2
print NN.forward(X)'''
print 

'''