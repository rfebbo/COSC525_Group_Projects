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
    def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.layers = []
        for i in range(numOfLayers):
            if weights is None:
                self.layers.append(Layers.FullyConnected(numOfNeurons[i], activation, inputSize, lr))
            else:
                self.layers.append(Layers.FullyConnected(numOfNeurons[i], activation, inputSize, lr, weights[i]))

        self.loss = loss
        self.activation = activation
        
        # print('constructor') 
    
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
        # do forward pass
        self.calculate(x)

        # calculate total error
        self.e_total = self.calculateloss(self.out[-1], y)

        # calculate d_error for last layer
        d_error = self.lossderiv(self.out[-1], y)

        # calculate delta for last layer Update weights?
        delta = self.layers[-1].calcwdeltas(d_error)
        # d_out_d_net = []
        # for n in self.layers[-1].neurons:
        #     d_out_d_net.append(n.activationderivative())
        
        # delta = d_error * d_out_d_net

        # update weights using delta
        for i, l in enumerate(reversed(self.layers)):
            if i == 0:
                continue
            delta = l.calcwdeltas(delta)

        # print (self.out[-1])
        return self.out[-1]

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        y=np.array([0.01,0.99])
        num_neurons = [2, 2]
        n = NeuralNetwork(2, num_neurons, len(x), 1, 0, 0.5, w)
        for i in range(2000):
            yp = n.train(x, y)
            if(i % 100):
                print("interation:", i)
                print("output: ", yp, y)
                print("Error", n.e_total)
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        x=[[0, 0], [0, 1], [1, 0], [1, 1]]
        y=[0, 1, 1, 0]
        n = NeuralNetwork(2, 2, 2, 1, 0, 0.1)

        for e in range(1000):
            for i in range(len(x)):
                yp = n.train(x[i], y[i])

            
            if(e % 100):
                print("interation:", i)
                print("output: ", yp, y)
                print("Error", n.e_total)
            
        print('learn xor')