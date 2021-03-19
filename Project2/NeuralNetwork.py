import numpy as np
import sys
from mymath import convolve_2d
from ConvolutionLayer import ConvolutionalLayer
from FullyConnected import FullyConnected
from Neuron import Neuron
import example1Test as EX1
import example2Test as EX2
import example3Test as EX3
import example4Test as EX4
from tensorflowtest_example1 import run_tf_example1
from tensorflowtest_example2 import run_tf_example2
from tensorflowtest_example3 import run_tf_example3
from tensorflowtest_example4 import run_tf_example4
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']="" 

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""

class MaxPoolingLayer:
    def __init__(self, kernelSize, inputDim, stride=None, name=None):
        #print("Max Pooling Layer")
        self.kernelSize = kernelSize
        self.stride = stride
        if(stride is None):
            self.stride = kernelSize
        self.name=name
        self.inputDim = inputDim
        self.outputShape = (inputDim[0], int((self.inputDim[1]-self.kernelSize)/self.stride) + 1, int((self.inputDim[2]-self.kernelSize)/self.stride) + 1)
        self.out = []

    def hasWeights(self):
        return False
        
        
    def calculate(self, inpu):     
        self.mLoc = []
        self.out = []
        inp = inpu[0] #get rid of extra dim that comes from conv layers

        self.out = np.zeros(self.outputShape)
        for c in range(self.inputDim[0]):
            mLocChannel = []
            # self.out.append((np.zeros((int((self.inputDim[1]-self.kernelSize)/self.stride) + 1, int((self.inputDim[2]-self.kernelSize)/self.stride) + 1))))
            
            for i in range(self.out.shape[1]):
                for j in range(self.out.shape[2]):
                    mLoc = []
                    m = []
                    for k in range(self.kernelSize):
                        for l in range(self.kernelSize):
                            #print("inp[",c,"][",i+k,"][",j+l,"]")
                            m.append(inp[c][i*self.stride+k][j*self.stride+l])
                            mLoc.append((i*self.stride+k, j*self.stride+l))
                    self.out[c,i,j] = max(m)            
                    mLocChannel.append(mLoc[np.argmax(m)])
            self.mLoc.append(mLocChannel)
        # print(self.mLoc)
        # print(self.out)
        return self.out
        
     
    # wd should be in 3d to include multiple channels
    def calcwdeltas(self, wd):
        self.newwd = np.zeros((self.inputDim[0], self.inputDim[1], self.inputDim[2]))
        for c in range(len(self.mLoc)):
            flat = np.asarray(wd[0]).flatten()
            for i in range(len(flat)):
                self.newwd[c][self.mLoc[c][i][0]][self.mLoc[c][i][1]] = flat[i]
        return self.newwd
        
class FlattenLayer:
    def __init__(self, inputDim, name=None):
        self.inputDim = inputDim
        self.name=name
        self.outputShape = np.size(np.zeros(inputDim))
        
    def calculate(self, i):
        self.out = (np.asarray(i).flatten())
        return self.out
    
    def hasWeights(self):
        return False

    def calcwdeltas(self, wd):
        return (np.reshape(wd, (self.inputDim[0], self.inputDim[1], self.inputDim[2])))
            

#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, inputSize, loss, lr):
        self.layers = []
        self.loss = loss
        self.inputSize = inputSize
        self.last_outputShape = inputSize
        self.lr = lr

    # add a layer to the neural network
    # input with layer type (FullyConnected, ConvolutionalLayer, MaxPoolingLayer, FlattenLayer)
    # call using keyword parameters
    def addLayer(self, layerType, numOfNeurons=None, activation=None, input_num = None, weights=None, numKernels=None, kernelSize=None, inputDim=None, name=None, padding=0, stride=None):
        #check if we have a layer before so we can use it's input shape
        if self.last_outputShape is not None:
            inputDim = self.last_outputShape
            # print("last shape = ", inputDim)

        if layerType == "FullyConnected":
            if weights is None:
                self.layers.append(FullyConnected(numOfNeurons, activation, inputDim, self.lr, name=name))
            else:
                self.layers.append(FullyConnected(numOfNeurons, activation, inputDim, self.lr, weights, name=name))
        elif layerType == "ConvolutionLayer":
            if weights is None:
                self.layers.append(ConvolutionalLayer(numKernels, kernelSize, activation, inputDim, self.lr, name=name, padding=padding))
            else:
                self.layers.append(ConvolutionalLayer(numKernels, kernelSize, activation, inputDim, self.lr, weights=weights, name=name, padding=padding))
        elif layerType == "MaxPoolingLayer":
            self.layers.append(MaxPoolingLayer(kernelSize, inputDim, stride=stride, name=name))
        elif layerType == "FlattenLayer":
            self.layers.append(FlattenLayer(inputDim, name=name))
        else:
            print("layerType must FullyConnected, ConvolutionalLayer, MaxPoolingLayer, or FlattenLayer")
            sys.exit()

        self.last_outputShape = self.layers[-1].outputShape

    #predict output for given input
    def predict(self, input):
        self.calculate(input)
        return self.out[-1] 
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        self.out = [] #set outputs of each layer to empty list
        self.out.append(self.layers[0].calculate(input))
        # print("output",self.out)
        for i, l in enumerate(self.layers):
            if i == 0:
                continue
            self.out.append(l.calculate(self.out[-1]))

        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        if self.loss == 'MSE': #mean squared error
            return (1 / y.size) * np.sum(((yp - y))**2)
        elif self.loss == 'SCCE': #sparse categorical cross entropy
            return -np.mean(y*np.log(y) + (1-yp)*np.log(1-y))
        else:
            print(f'Unknown Loss Funtion {self.loss}.')
            exit()
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 'MSE': #mean squared error
            return 2 * (yp - y)
        elif self.loss == 'SCCE': #sparse categorical cross entropy
            return -y/yp + (1-y)/(1-yp)
        else:
            print(f'Unknown Loss Funtion {self.loss}.')
            exit()

    def update_weights(self):
        for l in self.layers:
            l.update_weights()
    
    #Given a single input and desired output preform one step of backpropagation
    # (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        # do forward pass
        self.calculate(x)

        # calculate total error
        self.e_total = self.calculateloss(self.out[-1], y)

        # calculate d_error for last layer
        d_error = np.transpose(self.lossderiv(self.out[-1], y))
        # print('d_error shape: ', d_error.shape)
        # # calculate delta and propogate it back
        wdelta = []

        for i, l in enumerate(reversed(self.layers)):
            if i == 0: #the last layer gets d_error
                wdelta = l.calcwdeltas(d_error)
            else:
                wdelta = l.calcwdeltas(wdelta)

        return self.out[-1]

if __name__=="__main__":
    if (len(sys.argv)<2):
        #print('a good place to test different parts of your code')
        print('Please provide an argument: "example1", "example2"... up to "example4"')
        exit()

    verbose = False

    if (len(sys.argv) == 3):
        if(sys.argv[2].lower() == "true"):
            verbose = True

    if (sys.argv[1]=='example1'):
        labels = ['conv3 kernel1', 'conv3 kernel1 bias', 'FC weights', 'FC bias']
        #run our example code and TF's example code
        w = EX1.run_example1(verbose=verbose)
        w_test = run_tf_example1(verbose = verbose)

        w = list(w)
        w_test = list(w_test)
        
        #loop through layer weights
        for i, l_w in enumerate(w):
            w_test_np = np.asarray(w_test[i])
            l_w_np = np.asarray(l_w)
            mse = (1 / w_test_np.size) * np.sum(((l_w_np - w_test_np))**2)
            print(labels[i], 'mse: ', mse)
        
    elif(sys.argv[1]=='example2'):
        labels = ['conv3_1 kernel', 'conv3_1 kernel bias', 'conv3_2 kernel', 'conv3_2 kernel bias', 'FC weights', 'FC bias']
        w = EX2.run_example2(verbose=verbose)
        w_test = run_tf_example2(verbose = verbose)

        w = list(w)
        w_test = list(w_test)

        for i, l_w in enumerate(w):
            w_test_np = np.asarray(w_test[i])
            l_w_np = np.asarray(l_w)
            mse = (1 / w_test_np.size) * np.sum(((l_w_np - w_test_np))**2)
            print(labels[i], 'mse: ', mse)
        
    elif(sys.argv[1]=='example3'):
        labels = ['conv3_1 kernel', 'conv3_1 kernel bias', 'FC weights', 'FC bias']
        w = EX3.run_example3(verbose=verbose)
        w_test = run_tf_example3(verbose = verbose)

        w = list(w)
        w_test = list(w_test)

        #loop through layer weights
        for i, l_w in enumerate(w):
            w_test_np = np.asarray(w_test[i])
            l_w_np = np.asarray(l_w)
            mse = (1 / w_test_np.size) * np.sum(((l_w_np - w_test_np))**2)
            print(labels[i], 'mse: ', mse)
    elif(sys.argv[1]=='example4'):
        labels = ['conv3_1 kernel', 'conv3_1 kernel bias', 'FC 1 weights', 'FC 1 bias', 'FC 2 weights', 'FC 2 bias']
        w = EX4.run_example4(verbose=verbose)
        w_test = run_tf_example4(verbose = verbose)

        w = list(w)
        w_test = list(w_test)

        #loop through layer weights
        for i, l_w in enumerate(w):
            w_test_np = np.asarray(w_test[i])
            l_w_np = np.asarray(l_w)
            mse = (1 / w_test_np.size) * np.sum(((l_w_np - w_test_np))**2)
            print(labels[i], 'mse: ', mse)
    else:
        print('Please provide an argument: "example1", "example2"... up to "example4"')