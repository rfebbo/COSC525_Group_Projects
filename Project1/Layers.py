import numpy as np
import neuron

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        if weights is None:
            self.neurons = [neuron.Neuron(activation, input_num, lr) for i in range(numOfNeurons)]
        else:
            self.neurons = [neuron.Neuron(activation, input_num, lr, weights[i]) for i in range(numOfNeurons)]
        self.lr = lr
        
        # print('constructor') 
        
        
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
        # print("wtimesdelta ", wtimesdelta)
        for i, n in enumerate(self.neurons):
            # print(i)
            s = np.sum(n.weights * wtimesdelta) 
            w_delta.append(s * n.calcpartialderivative(wtimesdelta[i]))
            n.updateweight()

        return np.squeeze(w_delta)
        # print('calcwdeltas')
           

# f = FullyConnected(8, 0, 2, 0.1)
