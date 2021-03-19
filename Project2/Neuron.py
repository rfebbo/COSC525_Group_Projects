import numpy as np
import sys

"""
Notes
For each neuron, the bias is stored as the last weight of the weight array, so the length of the weight array should be input_nums + 1
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    # Bias = last weight of weight array
    # len(weights) should = intput_num + 1
    def __init__(self,activation, input_num, lr, weights=None):
        #print('constructor')    
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        # print('got weights: ', weights)
        # determine weights either randomly or with inputs
        if weights is None:
           self.weights = np.random.rand(input_num)
           self.bias = float(np.random.rand(1))
        elif len(weights[0]) == input_num:
            self.bias = weights[1]
            self.weights = np.asarray(weights[0])
            # print(self.bias)
        else:
            print('Neuron:')
            print('expected: ', input_num)
            print('got: ', self.weights.shape)

            # print("len(weights) = input_num + 1")
            sys.exit()
           
        
       
    #This method returns the activation of the net
    # def activate(self,net):

    #     if self.activation == 'linear':
    #         self.out = self.net
    #     elif self.activation == 'sigmoid':
    #         self.out = 1 / (1 + np.exp(-self.net))
    #     elif self.activation.lower() == 'relu':
    #         self.out = np.fmax(0, self.net)
    #     else:
    #         print(f'Unknown Activation Function {self.activation}')
    #         exit()

    #     return self.out
    
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        #print('calculate')
        input = np.asarray(input)
        if len(input) != self.input_num:
            print("len(input) = input_num")
            print(input)
            print(input.shape)
            print(self.input_num)
            sys.exit()
        self.input = input
        self.net = np.dot(self.input,self.weights) + self.bias
        return self.net
        

    # #This method returns the derivative of the activation function with respect to the net   
    # def activationderivative(self):

    #     if self.activation == 'linear':
    #         self.dactive = self.out
    #     elif self.activation == 'sigmoid':
    #         self.dactive = self.out * (1 - self.out)
    #     elif self.activation.lower() == 'relu':
    #         self.dactive = (self.net > 0) * 1
    #     else:
    #         print(f'Unknown Activation Function {self.activation}')
    #         exit()

    #     return self.dactive
        
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta, dactive):
        # self.activationderivative()

        self.delta = wtimesdelta * dactive
        self.d_error = self.delta * self.input

        return self.delta * self.weights
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        # print('updateweight')
        self.weights = self.weights - (self.lr * self.d_error)
        self.bias = self.bias - (self.lr * self.delta)
        # print(self.bias)