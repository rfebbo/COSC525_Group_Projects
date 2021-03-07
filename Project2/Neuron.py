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
        self.activation = activation;
        self.input_num = input_num;
        self.lr = lr;
        # determine weights either randomly or with inputs
        if weights is None:
           self.weights = np.random.rand(input_num);
           self.bias = float(np.random.rand(1));
        elif len(weights) == input_num + 1:
            self.bias = weights[-1];
            self.weights = np.asarray(weights[:-1]);
            # print(self.bias)
        else:
            print(input_num)
            print(weights.shape)

            # print("len(weights) = input_num + 1")
            sys.exit();
           
        
       
    #This method returns the activation of the net
    def activate(self,net):
        #print('activate')   
        # linear
        # f(x) = x
        if self.activation == 0:
            self.out = self.net;
        
        # logistic
        # f(x) = 1 / (1 + e^(-x))
        else:
            #print("logistic")
            self.out = 1 / (1 + np.exp(-net))

        return self.out
    
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        #print('calculate')
        input = np.asarray(input)
        if len(input) != self.input_num:
            print("len(input) = input_num")
            print(input)
            print(input.shape)
            print(self.input_num)
            sys.exit();
        self.input = input;
        self.net = np.dot(self.input,self.weights) + self.bias;
        return self.net
        

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        #print('activationderivative')
        # linear
        # df(x) = 1
        if(self.activation == 0):
            self.dactive = self.out;
        
        # logistic
        # df(x) = out(1 - out)
        else:
            self.dactive = self.out * (1 - self.out);

        return self.dactive
        
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        self.activationderivative()

        self.delta = wtimesdelta * self.dactive
        self.d_error = self.delta * self.input

        return self.delta * self.weights
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        # print('updateweight')
        self.weights = self.weights - (self.lr * self.d_error);
        self.bias = self.bias - (self.lr * self.delta);
        # print(self.bias)