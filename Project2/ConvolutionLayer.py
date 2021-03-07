import numpy as np
import sys
from mymath import convolve_2d
from Neuron import Neuron

class ConvolutionalLayer:
    def __init__(self, numKernels, kernelSize, activation, inputDim, lr, weights=None, name=None):
        # print("Convolutional Layer")
        self.numKernels = numKernels
        self.name=name
        self.kernelSize = kernelSize
        self.activation = activation
        if(len(inputDim) == 3):
            inputDim = (1, inputDim[0], inputDim[1], inputDim[2])
        self.inputDim = inputDim
        self.lr = lr
        self.weightsShape = (numKernels, inputDim[1], kernelSize[0], kernelSize[1])

        #initialize weights
        if weights is None:
            self.weights = np.random.random_sample(self.weightsShape)
            self.bias = [float(np.random.rand(numKernels))]
        else: #removed error checking!
            self.bias = weights[-1]
            self.weights = np.asarray(weights[:-1]).reshape((self.weightsShape))

        # shorthand for filter shape sizes
        Nf = self.weights.shape[0]
        Cf = self.weights.shape[1]
        Hf = self.weights.shape[2]
        Wf = self.weights.shape[3]

        if (Cf != inputDim[1]):
            print(f'number of channels in filter {Cf} does not match input {inputDim[1]}')
            sys.exit()

        # determine output shape size
        self.padding = 0
        self.stride = 1
        No = inputDim[0]
        Co = Nf
        Ho = int((inputDim[2] - Hf + self.padding * 2 + self.stride)  / self.stride)
        Wo = int((inputDim[3] - Wf + self.padding * 2 + self.stride) / self.stride)
        self.outputShape = (Co, Ho, Wo)

        self.neurons = [None] * Co

        # initialize neurons
        # for c_o in range(Co):
        #     self.neurons[c_o] = [None] * Ho
        #     for h_o in range(Ho):
        #         self.neurons[c_o][h_o] = [None] * Wo
        #         for w_o in range(Wo):
        #             self.neurons[c_o][h_o][w_o] = Neuron(activation, (Cf,Hf,Wf), lr, self.weights[c_o,:,:,:])
                    


#calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.input = input
        self.net = convolve_2d(input, self.weights, self.bias, self.stride, self.padding)
        
        if (self.activation == 1):
            self.out = 1 / (1 + np.exp(-self.net))
            self.dactive = self.out * (1 - self.out)
        else:
            self.dactive = self.out = self.net

        return self.out
        
    def update_weights(self):
        pass
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for 
    # each (with the correct value), sum up its ownw*delta (just delta?), 
    # and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        # calculate deltas
        self.delta = wtimesdelta * self.dactive

        self.delta = np.sum(self.delta, axis=1).reshape(1,1,self.out.shape[2], self.out.shape[3])

        # print("delta shape: ", self.delta.shape)
        for c in range(self.input.shape[1]): #why do the other channels on the input not seem to matter?
            self.d_error_w = convolve_2d(self.input[:,c,:,:].reshape((1,1, self.input.shape[2], self.input.shape[3])), self.delta , np.zeros_like(self.bias), self.stride, self.padding)


        self.d_error_w = np.asarray(self.d_error_w)
        #update weights and bias
        self.weights = self.weights - (self.lr * self.d_error_w)
        self.bias = self.bias - (self.lr * np.sum(self.delta))


        # print("weights updated")
        #calculate w times delta for this layer
        for n in range(self.weights.shape[0]): #why do these dims not seem to matter either?
            for c in range(self.weights.shape[1]):
                wtimesdelta = convolve_2d(self.delta, np.flip(self.weights[n,c,:,:].reshape((1,1, self.weightsShape[2], self.weightsShape[3]))), np.zeros_like(self.bias), self.stride, 2)

        # print("wtimesdelta: ", wtimesdelta.shape)
        return wtimesdelta
