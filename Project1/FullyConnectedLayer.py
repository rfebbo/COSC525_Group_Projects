import numpy as np
import neuron

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.neurons = [neuron.Neuron(activation, input_num, lr, weights) for i in range(numOfNeurons)]
        self.lr = lr
        
        print('constructor') 
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.out = []
        for n in (self.neurons):
            self.out.append(n.activate(n.calculate(input)))

        return self.out
        # print('calculate') 
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for 
    # each (with the correct value), sum up its ownw*delta, 
    # and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        for n, i in enumerate(self.neurons):
            n.calcpartialderivative(wtimesdelta[i])
            n.updateweight()
        print('calcwdeltas')
           

f = FullyConnected(8, 0, 2, 0.1)
