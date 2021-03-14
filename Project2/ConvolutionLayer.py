import numpy as np
import sys
from mymath import convolve_2d
from Neuron import Neuron

class ConvolutionalLayer:
    def __init__(self, numKernels, kernelSize, activation, inputDim, lr, padding=0, stride=1, weights=None, name=None):
        # print("Convolutional Layer")
        self.numKernels = numKernels
        self.name=name
        self.kernelSize = kernelSize
        self.activation = activation

        if(padding is None):
            self.padding = 0
        else:
            self.padding = padding

        if(stride is None):
            self.stride = stride
        else:
            self.stride = stride


        print("input dim: ", inputDim)
        if(len(inputDim) == 3): # the conv code is setup for 4D for batches but nothing else is so this needs to be checked
            inputDim = (1, inputDim[0], inputDim[1], inputDim[2])
        self.inputDim = inputDim
        self.lr = lr
        self.weightsShape = (numKernels, inputDim[1], kernelSize[0], kernelSize[1])

        #initialize weights
        if weights is None:
            self.weights = np.random.random_sample(self.weightsShape)
            self.bias = [(np.random.rand(numKernels))]
        else: #removed error checking!
            self.bias = weights[-1]
            self.weights = np.asarray(weights[:-1]).reshape((self.weightsShape))

        # shorthand for filter shape sizes
        Nf = self.weights.shape[0]
        Cf = self.weights.shape[1]
        Hf = self.weights.shape[2]
        Wf = self.weights.shape[3]

        #check we have a filter of correct dim size(shape size)
        if (Cf != inputDim[1]):
            print(f'number of channels in filter {Cf} does not match input {inputDim[1]}')
            sys.exit()

        # determine output shape size
        print('hf: ', Hf)
        print('padding: ', self.padding)
        print('stride: ', self.stride)
        print('inputDim[2]: ', self.inputDim[2])
        self.padding = padding
        self.stride = stride
        No = self.inputDim[0]
        Co = Nf
        Ho = int((self.inputDim[2] - Hf + self.padding * 2 + self.stride)  / self.stride)
        Wo = int((self.inputDim[3] - Wf + self.padding * 2 + self.stride) / self.stride)
        self.outputShape = (Co, Ho, Wo)
 
        #we didn't end up using Neuron objects :(
        # self.neurons = [None] * Co

        # initialize neurons
        # for c_o in range(Co):
        #     self.neurons[c_o] = [None] * Ho
        #     for h_o in range(Ho):
        #         self.neurons[c_o][h_o] = [None] * Wo
        #         for w_o in range(Wo):
        #             self.neurons[c_o][h_o][w_o] = Neuron(activation, (Cf,Hf,Wf), lr, self.weights[c_o,:,:,:])
                    


#calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.input = input.reshape(self.inputDim)
        

        #use custom conv function to do convolution and get net
        self.net = convolve_2d(self.input, self.weights, self.bias, self.stride, self.padding)
        
        #calculate out and dactive depending on activation
        if (self.activation == 'sigmoid'):
            self.out = 1 / (1 + np.exp(-self.net))
            self.dactive = self.out * (1 - self.out)
        elif self.activation == 'linear':
            self.dactive = self.out = self.net
        elif self.activation.lower() == 'relu':
            self.out = np.fmax(0, self.net)
            self.dactive = (self.net > 0) * 1
        else:
            print(f'Unknown Activation Function {self.activation}')
            exit()

        return self.out
        
    def update_weights(self):
        pass

    def hasWeights(self):
        return True
            
    #given the next layer's w*delta, uses conv2d function to calculate the delta
    # of this layers neurons, and to calculate wtimesdelta to send back to the previous layer. 
    # padding and flipped weights are employed to get the wtimesdelta
    def calcwdeltas(self, wtimesdelta):
        # calculate deltas
        self.delta = wtimesdelta * self.dactive

        #delta can come back with an extra dim sometimes so it is summed here
        self.delta = np.sum(self.delta, axis=1).reshape(1,1,self.out.shape[2], self.out.shape[3])

        #kind of lost here. we are not using all of the input channels. d_error_w is getting overwritten every loop...
        for c in range(self.input.shape[1]): #why do the other channels on the input not seem to matter?
            self.d_error_w = convolve_2d(self.input[:,c,:,:].reshape((1,1, self.input.shape[2], self.input.shape[3])), self.delta , np.zeros_like(self.bias), self.stride, self.padding)

        self.d_error_w = np.asarray(self.d_error_w)
        
        #update weights and bias
        self.weights = self.weights - (self.lr * self.d_error_w)
        self.bias = self.bias - (self.lr * np.sum(self.delta))

        #calculate w times delta for this layer
        for n in range(self.weights.shape[0]): #why do these dims not seem to matter either?
            for c in range(self.weights.shape[1]): #wtimesdelta is getting overwritten every loop... we only need one?
                wtimesdelta = convolve_2d(self.delta, np.flip(self.weights[n,c,:,:].reshape((1,1, self.weightsShape[2], self.weightsShape[3]))), np.zeros_like(self.bias), self.stride, 2)

        return wtimesdelta
