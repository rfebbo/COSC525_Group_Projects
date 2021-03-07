import numpy as np
from Neuron import Neuron

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None, name=None):
        self.input_num = input_num
        self.lr = lr
        self.name=name
        self.num_neurons = numOfNeurons
        if weights is None:
            self.neurons = [Neuron(activation, input_num, lr) for i in range(numOfNeurons)]
        else:
            self.neurons = [Neuron(activation, input_num, lr, weights[i]) for i in range(numOfNeurons)]
        
        self.outputShape = numOfNeurons
        self.update_weights()
        
        # print('constructor') 
    
    #function to pull the weights from the neurons so we can compare them with TF
    def update_weights(self):
        self.weights = [None] * self.num_neurons
        self.bias = [None] * self.num_neurons
        for i, n in enumerate(self.neurons):
            self.weights[i] = n.weights
            self.bias[i] = n.bias
        self.weights = np.asarray(self.weights)
        self.bias = np.asarray(self.bias)
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.out = []
        # print("input",input)
        for n in (self.neurons):
            n_output = n.activate(n.calculate(input))
            n.activationderivative()
            # print("n_output",n_output)
            self.out.append(n_output)

        return self.out
        # print('calculate') 
        
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for 
    # each (with the correct value), sum up its ownw*delta (just delta?), 
    # and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        w_delta = []

        for i, n in enumerate(self.neurons):
            w_delta.append(n.calcpartialderivative(wtimesdelta[i]))
            
            # print("neuron: ", i, " weights: ", n.weights, "bias: ", n.bias)
            n.updateweight()

        self.update_weights()
        w_delta = np.sum(w_delta, axis=0)
        return w_delta

        