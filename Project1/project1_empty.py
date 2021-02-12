import numpy as np
import sys
import Layers
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""

        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.layers = []
        for i in range(numOfLayers):
            self.layers.append(Layers.FullyConnected(numOfNeurons, activation, inputSize, lr, weights[i]))

        self.loss = loss
        self.activation = activation
        
        # print('constructor') 
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        self.out = []
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
            return 0.5 * np.sum(((yp - y))**2)
        else:
            return -np.mean(y*np.log(y) + (1-yp)*np.log(1-y))
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0:
            return -(y - yp)
        else:
            return -y/yp + (1-y)/(1-yp)
    
    #Given a single input and desired output preform one step of backpropagation
    # (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        
        self.calculate(x)
        self.e_total = self.calculateloss(self.out[-1], y)

        d_error = self.lossderiv(self.out[-1], y)

        d_out_d_net = []
        for n in self.layers[-1].neurons:
            d_out_d_net.append(n.activationderivative())
        
        delta = d_error * d_out_d_net

        for i, l in enumerate(reversed(self.layers)):
            if i == 0:
            # if i == len(self.layers) - 1:
                continue
            delta = l.calcwdeltas(delta)


        # print(delta)
        return self.out[-1]
        # print('train')

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        y=np.array([0.01,0.99])
        n = NeuralNetwork(2, 2, len(x), 1, 0, 0.01, w)
        for i in range(1000):
            yp = n.train(x, y)
            if(i % 100):
                print(0.5 * np.sum(((yp - y))**2))
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')