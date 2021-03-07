import numpy as np
import sys
from mymath import convolve_2d
from ConvolutionLayer import ConvolutionalLayer
from FullyConnected import FullyConnected
from Neuron import Neuron
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
    def __init__(self, kernelSize, inputDim, name=None):
        #print("Max Pooling Layer")
        self.kernelSize = kernelSize;
        self.name=name
        self.inputDim = inputDim;
        self.out = [];
        
    def calculate(self, inp):     
        self.mLoc = []
        for c in range(self.inputDim[0]):
            mLocChannel = []
            self.out.append((np.zeros((int(self.inputDim[1]/self.kernelSize), int(self.inputDim[2]/self.kernelSize)))))
            for i in range(len(self.out[0])):
                for j in range(len(self.out[c][0])):
                    mLoc = []
                    m = []
                    for k in range(self.kernelSize):
                        for l in range(self.kernelSize):
                            #print("inp[",c,"][",i+k,"][",j+l,"]")
                            m.append(inp[c][i*self.kernelSize+k][j*self.kernelSize+l])
                            mLoc.append((i*self.kernelSize+k, j*self.kernelSize+l))
                    self.out[c][i][j] = max(m)            
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
                self.newwd[c][self.mLoc[c][i][0]][self.mLoc[c][i][1]] = flat[i];
        return self.newwd
        
class FlattenLayer:
    def __init__(self, inputDim, name=None):
        self.inputDim = inputDim;
        self.name=name
        
    def calculate(self, i):
        self.out = (np.asarray(i).flatten())
        return self.out
    
    def calcwdeltas(self, wd):
        return (np.reshape(wd, (self.inputDim[0], self.inputDim[1], self.inputDim[2])))
            

#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, inputSize, loss, lr):
        """def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.layers = []
        for i in range(numOfLayers):
            if weights is None:
                self.layers.append(FullyConnected(numOfNeurons[i], activation, inputSize, lr))
            else:
                self.layers.append(FullyConnected(numOfNeurons[i], activation, inputSize, lr, weights[i]))
                """
        self.layers = []
        self.loss = loss
        self.inputSize = inputSize
        self.lr = lr;
        #self.activation = activation
        
        # print('constructor') 

    # add a layer to the neural network
    # input with layer type (FullyConnected, ConvolutionalLayer, MaxPoolingLayer, FlattenLayer)
    # call using keyword parameters
    def addLayer(self, layerType, numOfNeurons=None, activation=None, input_num = None, weights=None, numKernels=None, kernelSize=None, inputDim=None, name=None):
        # if len(self.layers) != 0:
        #     self.inputSize = (len(self.layers[-1].out),len(self.layers[0][0]),len(self.layers[0]))
        # print(self.inputSize)
        if layerType == "FullyConnected":
            # print("FullyConnected")
            if weights is None:
                self.layers.append(FullyConnected(numOfNeurons, activation, input_num, self.lr, name=name))
            else:
                self.layers.append(FullyConnected(numOfNeurons, activation, input_num, self.lr, weights, name))
        elif layerType == "ConvolutionLayer":
            # print("ConvolutionalLayer")
            if weights is None:
                self.layers.append(ConvolutionalLayer(numKernels, kernelSize, activation, inputDim, self.lr, name=name));
            else:
                self.layers.append(ConvolutionalLayer(numKernels, kernelSize, activation, inputDim, self.lr, weights, name));
        elif layerType == "MaxPoolingLayer":
            #print("MaxPoolingLayer")
            self.layers.append(MaxPoolingLayer(kernelSize, inputDim, name=name));
        elif layerType == "FlattenLayer":
            # change to inputDim
            # self.layers.append(FlattenLayer(self.inputSize));
            self.layers.append(FlattenLayer(inputDim, name=name));
        else:
            print("layerType must FullyConnected, ConvolutionalLayer, MaxPoolingLayer, or FlattenLayer")
            sys.exit();

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
            # print("output",self.out)
            self.out.append(l.calculate(self.out[-1]))

        # print(self.out)

        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        if self.loss == 0:
            return (1 / len(y)) * np.sum(((yp - y))**2)
        else:
            return -np.mean(y*np.log(y) + (1-yp)*np.log(1-y))
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0:
            return -2 * (y - yp)
        else:
            return -y/yp + (1-y)/(1-yp)

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
        d_error = self.lossderiv(self.out[-1], y)

        # calculate delta for last layer
        wdelta = []
        for i, n in enumerate(self.layers[-1].neurons):
            n.calcpartialderivative(d_error[i])
            wdelta_i = (n.weights * n.delta)
            wdelta.append(wdelta_i)
            
            # print("neuron: ", i, " weights: ", n.weights, "bias: ", n.bias)
            n.updateweight()

        wdelta = np.sum(wdelta, axis=0)
        
        # update weights using delta
        for i, l in enumerate(reversed(self.layers)):
            if i == 0:
                continue

            wdelta = l.calcwdeltas(wdelta)

        return self.out[-1]

if __name__=="__main__":
    if (len(sys.argv)<2):
        #print('a good place to test different parts of your code')
        N = NeuralNetwork([1,4,4], 1, 0.1);
        x = [[[1,2,3,4],[8,  2,9,10],[11,3,8,0],[0,1,4,7]]]
        #x = [[[1,2,3,4],[8,  2,9,10],[11,3,8,0],[0,1,4,7]],[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]]
        w = [[[8,10],[11,8]]]
        #N.addLayer("FullyConnected", activation=0, numOfNeurons=2, input_num=2)

        #N.addLayer("ConvolutionalLayer", numKernels = 1, kernelSize=3, activation=1, inputDim=[2,2, 1])


        N.addLayer("MaxPoolingLayer", kernelSize=2, inputDim=[1,4,4])
        N.layers[0].calculate(x)
        N.layers[0].calculatewdeltas(w)
        
        N.addLayer("FlattenLayer", inputDim=[1,4,4])
        flat = N.layers[1].calculate(x)
        N.layers[1].calculatewdeltas(flat)

    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        y=np.array([0.01,0.99])
        num_neurons = [2, 2]
        n = NeuralNetwork(len(x), 0, 0.5)
        #n = NeuralNetwork(2, num_neurons, len(x), 1, 0, 0.5, w)
        for i in range(1001):
            yp = n.train(x, y)
            if(i % 100 == 0):
                print("interation:", i)
                print("output: ", yp, y)
                print("Error", n.e_total)
                print()
        
    elif(sys.argv[1]=='and'):
        x=[[0, 0], [0, 1], [1, 0], [1, 1]]
        y=[0.0, 0.0, 0.0, 1.0]
        y = np.asarray(y)
        num_neurons = [1]
        n = NeuralNetwork(1, num_neurons, 2, 1, 0, 0.1)

        for e in range(2001):
            for i in range(len(x)):
                yp = n.train(x[i], y[i])

            if(e % 100 == 0):
                print("interation:", e)
                
                for i in range(len(x)):
                    print("sample", i)
                    yp = n.train(x[i], y[i])
                    print("\tintput: ", x[i])
                    print("\toutput: ", yp)
                    print("\tground truth: ", y[i])
                
                print("Error", n.e_total)
        
    elif(sys.argv[1]=='xor'):
        x=[[0, 0], [0, 1], [1, 0], [1, 1]]
        y=[0.0, 1.0, 1.0, 0.0]
        y = np.asarray(y)
        num_neurons = [2, 1]
        n = NeuralNetwork(2, num_neurons, 2, 1, 0, 0.1)

        for e in range(50001):
            for i in range(len(x)):
                yp = n.train(x[i], y[i])

            if(e % 1000 == 0):
                print("interation:", e)
                
                for i in range(len(x)):
                    print("sample", i)
                    yp = n.train(x[i], y[i])
                    print("\tintput: ", x[i])
                    print("\toutput: ", yp)
                    print("\tground truth: ", y[i])
                
                print("Error", n.e_total)