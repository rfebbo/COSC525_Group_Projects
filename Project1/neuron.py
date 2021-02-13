import numpy as np
import sys
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""

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
            print(self.bias)
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
            # print("len(input) = input_num")
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
        # print('calcpartialderivative')
        self.activationderivative()

        self.delta = wtimesdelta * self.dactive
        self.d_error = self.delta * self.input
        # not sure how to handle biases here
        # for w in self.weights:
        #     self.delta = wtimesdelta 
        return self.delta * self.weights 
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        # print('updateweight')
        self.weights = self.weights - (self.lr * self.d_error);
        self.bias = self.bias - (self.lr * self.delta);
        # print(self.bias)

"""
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        print('constructor') 
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        print('calculate') 
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        print('calcwdeltas') 
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        print('constructor') 
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        print('constructor')
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        print('calculate')
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        print('lossderiv')
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        print('train')

def ntest(n, i):
        n.calculate(i);
        n.activate(n.net)
        
        #print(n.output)
        #print(n.out)
        #print()
        

if __name__=="__main__":
    if (len(sys.argv)<2):
        #print('a good place to test different parts of your code')
        h1 = Neuron(1, 2, 0.5, [.15, .20, .35]);
        h2 = Neuron(1, 2, 0.5, [.25, .30, .35]);
        o1 = Neuron(1, 2, 0.5, [.40, .45, .60]);
        o2 = Neuron(1, 2, 0.5, [.50, .55, .60]);
        #print("h1")
        ntest(h1, [.05, .1])
        #print("h2")
        ntest(h2, [.05, .1])
        #print("o1")
        ntest(o1, [h1.out, h2.out]);
        #print("o2")
        ntest(o2, [h1.out, h2.out]);
        
        # Error
        e1 = 0.5 * (0.01 - o1.out)**2
        e2 = 0.5 * (0.99 - o2.out)**2
        et = e1 + e2;
        print(et)
        o1.activationderivative()
        
        #d
        
        print(o1.dactive)

        
        #n = Neuron(0, 4, .01)
        #n = Neuron(0, 4, .01, [.1, .2, .5, .1, .8])
        #print(n.weights)
        #print(n.bias)
        #n.calculate([1, 2, 3, 1])
        #print(n.net)
     
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        #x==np.array([0.05,0.1])
        np.array([0.01,0.99])
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')
"""
