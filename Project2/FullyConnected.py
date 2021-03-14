import numpy as np
from Neuron import Neuron

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None, name=None):
        self.input_num = input_num
        self.lr = lr
        self.name=name
        self.activation = activation
        self.num_neurons = numOfNeurons
        print('creating neurons for layer: ', name)
        if weights is None:
            self.neurons = [Neuron(activation, input_num, lr) for i in range(numOfNeurons)]
        else:
            # self.neurons = [Neuron(activation, input_num, lr, [weights[0][i,:],weights[1][i]]) for i in range(numOfNeurons)]
            self.neurons = [Neuron(activation, input_num, lr, [weights[0][:,i],weights[1][i]]) for i in range(numOfNeurons)]
        
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
        self.weights = np.transpose(np.asarray(self.weights))
        self.bias = np.asarray(self.bias)
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        self.net = np.empty((self.num_neurons))
        
        for i, n in enumerate(self.neurons):
            self.net[i] = n.calculate(input)

        if self.activation.lower() == 'linear':
            self.out = self.net
            self.dactive = self.net
        elif self.activation.lower() == 'sigmoid':
            self.out = 1 / (1 + np.exp(-self.net))
            self.dactive = self.out * (1 - self.out)
        elif self.activation.lower() == 'relu':
            self.out = np.fmax(0, self.net)
            self.dactive = (self.net > 0) * 1
        elif self.activation.lower() == 'softmax':
            exp = np.exp(self.net)
            self.out = exp / np.sum(exp)
            self.dactive = self.out * (1 - self.out)
        else:
            print(f'Unknown Activation Function {self.activation}')
            exit()

        return self.out
        
    def hasWeights(self):
        return True
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for 
    # each (with the correct value), sum up its ownw*delta (just delta?), 
    # and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        w_delta = []

        print('dactive shape: ', self.dactive.shape)
        print('wtimesdelta shape: ', wtimesdelta.shape)
        for i, n in enumerate(self.neurons):
            w_delta.append(n.calcpartialderivative(wtimesdelta[i], self.dactive[i]))
            n.updateweight()

        self.update_weights()
        w_delta = np.sum(w_delta, axis=0)
        return w_delta

        