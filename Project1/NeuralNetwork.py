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
        self.n_samples = 4

    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        self.out = [] #set outputs of each layer to empty list

        #calculate the output of the first layer
        self.out.append(self.layers[0].calculate(input))

        # calculate the output of the other layers
        for i, l in enumerate(self.layers):
            if i == 0:
                continue
            self.out.append(l.calculate(self.out[-1]))


        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        yp = np.array(yp)
        if self.loss == 0:
            return 0.5 * np.sum(((yp - y))**2)
        else:
            return (1/self.n_samples) * -(y * np.log(yp)) + (1 - y) * np.log((1- yp))
            # return -np.mean(y*np.log(y) + (1-yp)*np.log(1-y))
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        yp = np.array(yp)
        if self.loss == 0:
            return -(y - yp)
        else:
            return -y/yp + (1-y)/(1-yp)
    
    #Given a single input and desired output preform one step of backpropagation
    # (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y, verbose=False):
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
            if (verbose):
                print("Layer 2: neuron: ", i, " weights: ", n.weights, "bias: ", n.bias)
            n.updateweight()

        wdelta = np.sum(wdelta, axis=0)
        
        # update weights using delta and calculate other wdeltas for backprop
        l_n = len(self.layers) - 1 #for printing
        for i, l in enumerate(reversed(self.layers)):
            if i == 0:
                continue

            if (verbose):
                for j, n in enumerate(l.neurons):
                    print(f"Layer {l_n}: neuron: ", j, " weights: ", n.weights, "bias: ", n.bias)
                l_n -= 1
            wdelta = l.calcwdeltas(wdelta)
            
            

        return self.out[-1]

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('please provide an argument: "example", "and", or "xor"')
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        y=np.array([0.01,0.99])
        num_neurons = [2, 2] #neurons in each layer
        n = NeuralNetwork(2, num_neurons, len(x), 1, 0, 0.5, w)
        num_iters = 0
        for i in range(2):
            print("interation:", num_iters)
            yp = n.train(x, y, verbose=True) # prints weights before updating
            num_iters += 1
            print("output: ", yp, y)
            print("Error", n.e_total)
            print()

        # train for 1000 iterations to compare to in class results
        # for i in range(998):
        #     yp = n.train(x, y)
        #     num_iters += 1

        
        # print("interations:", num_iters)
        # print("output: ", yp, y)
        # print("Error", n.e_total)
        # print()
        
    elif(sys.argv[1]=='and'):
        #training samples
        x=[[0, 0], [0, 1], [1, 0], [1, 1]]
        #training labels
        y=[0.0, 0.0, 0.0, 1.0]
        y = np.asarray(y)
        num_neurons = [1]
        n = NeuralNetwork(1, num_neurons, 2, 1, 1, 0.1)

        for e in range(101):
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
        #training samples
        x=[[0, 0], [0, 1], [1, 0], [1, 1]]
        #training labels
        y=[0.0, 1.0, 1.0, 0.0]
        y = np.asarray(y)
        num_neurons = [2, 1]
        n = NeuralNetwork(2, num_neurons, 2, 1, 1, 0.1)

        for e in range(5001):
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