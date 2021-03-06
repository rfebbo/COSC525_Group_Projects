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
        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the learning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        if weights is None:
            self.neurons = [Neuron(activation, input_num, lr) for i in range(numOfNeurons)]
        else:
            self.neurons = [Neuron(activation, input_num, lr, weights[i]) for i in range(numOfNeurons)]
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

        for i, n in enumerate(self.neurons):
            w_delta.append(n.calcpartialderivative(wtimesdelta[i]))
            
            # print("neuron: ", i, " weights: ", n.weights, "bias: ", n.bias)
            n.updateweight()

        w_delta = np.sum(w_delta)
        return w_delta
        # print('calcwdeltas')

# input dim example [w, h, c]
#convolutional layer
"""
class ConvolutionalLayer:
    def __init__(self, numKernels, kernelSize, activation, inputDim, lr, weights=None):
        print("Convolutional Layer")
        self.numKernels = numKernels;
        self.kernelSize = kernelSize;
        self.activation = activation;
        self.inputDim = inputDim;
        self.lr = lr;
        if weights is None:
           self.weights = np.random.rand(inputDim[0]*inputDim[1]*inputDim[2]);
           self.bias = float(np.random.rand(1));
        elif len(weights) == inputDim[0] * inputDim[1]*inputDim[2]:
            self.bias = weights[-1];
            self.weights = np.asarray(weights[:-1]);
"""            
class MaxPoolingLayer:
    def __init__(self, kernelSize, inputDim):
        #print("Max Pooling Layer")
        self.kernelSize = kernelSize;
        self.inputDim = inputDim;
        self.pool = [];
        
    def calculate(self, inp):
        for c in range(self.inputDim[2]):
            self.pool.append((np.zeros((self.inputDim[0]-self.kernelSize+1, self.inputDim[1]-self.kernelSize+1))))
            for i in range(len(self.pool[0])):
                for j in range(len(self.pool[c][0])):
                    m = []
                    for k in range(self.kernelSize):
                        m += inp[c][i+k][j:j+self.kernelSize]
                    self.pool[c][i][j] = max(m)
        print(self.pool)
        
class FlattenLayer:
    def __init__(self, inputSize):
        print("Flatten Layer")
        

#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, inputSize, loss, lr):
        """def __init__(self, numOfLayers, numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.layers = []
        for i in range(numOfLayers):
            if weights is None:
                self.layers.append(FullyConnected(numOfNeurons[i], activation, inputSize, lr))
            else:
                self.layers.append(FullyConnected(numOfNeurons[i], activation, inputSize, lr, weights[i]))
                """
        self.layers = []
        self.loss = loss
        self.inputSize = inputSize
        self.lr = lr;
        #self.activation = activation
        
        # print('constructor') 

    # add a layer to the neural network
    # input with layer type (FullyConnected, ConvolutionalLayer, MaxPoolingLayer, FlattenLayer)
    # call using keyword parameters
    def addLayer(self, layerType, numOfNeurons=None, activation=None, input_num = None, weights=None, numKernels=None, kernelSize=None, inputDim=None):
        if layerType == "FullyConnected":
            print("FullyConnected")
            if weights is None:
                self.layers.append(FullyConnected(numOfNeurons, activation, input_num, self.lr))
            else:
                self.layers.append(FullyConnected(numOfNeurons, activation, input_num, self.lr, weights[i]))
        elif layerType == "ConvolutionalLayer":
            print("ConvolutionalLayer")
            if weights is None:
                self.layers.append(ConvolutionalLayer(numKernels, kernelSize, activation, inputDim, self.lr));
            else:
                self.layers.append(ConvolutionalLayer(numKernels, kernelSize, activation, inputDim, self.lr, weights));
        elif layerType == "MaxPoolingLayer":
            print("MaxPoolingLayer")
            self.layers.append(MaxPoolingLayer(kernelSize, inputDim));
        elif layerType == "FlattenLayer":
            self.layers.append(FlattenLayer(self.inputSize));
        else:
            print("layerType must FullyConnected, ConvolutionalLayer, MaxPoolingLayer, or FlattenLayer")
            sys.exit();
            
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        self.out = [] #set outputs of each layer to empty list
        self.out.append(self.layers[0].calculate(input))
        # print("output",self.out)
        for i, l in enumerate(self.layers):
            if i == 0:
                continue
            # print("output",self.out)
            self.out.append(l.calculate(self.out[-1]))

        # print(self.out)

        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        if self.loss == 0:
            return 0.5 * np.sum(((yp - y))**2)
        else:
            return -np.mean(y*np.log(y) + (1-yp)*np.log(1-y))
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0:
            return -(y - yp)
        else:
            return -y/yp + (1-y)/(1-yp)
    
    #Given a single input and desired output preform one step of backpropagation
    # (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
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
            
            # print("neuron: ", i, " weights: ", n.weights, "bias: ", n.bias)
            n.updateweight()

        wdelta = np.sum(wdelta, axis=0)
        
        # return
        # update weights using delta
        for i, l in enumerate(reversed(self.layers)):
            if i == 0:
                continue

            wdelta = l.calcwdeltas(wdelta)

        return self.out[-1]

if __name__=="__main__":
    if (len(sys.argv)<2):
        #print('a good place to test different parts of your code')
        N = NeuralNetwork(2, 1, 0.1);
        #N.addLayer("FullyConnected", activation=0, numOfNeurons=2, input_num=2)
        #N.addLayer("ConvolutionalLayer", numKernels = 1, kernelSize=3, activation=1, inputDim=[2,2, 1])
        N.addLayer("MaxPoolingLayer", kernelSize=2, inputDim=[4,4,1])
        x = [[[1,2,3,4],[8,2,9,10],[11,3,8,0],[0,1,4,7]]]
        N.layers[0].calculate(x)
        #N.addLayer("FlattenLayer")
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x=np.array([0.05,0.1])
        y=np.array([0.01,0.99])
        num_neurons = [2, 2]
        n = NeuralNetwork(len(x), 0, 0.5)
        #n = NeuralNetwork(2, num_neurons, len(x), 1, 0, 0.5, w)
        for i in range(1001):
            yp = n.train(x, y)
            if(i % 100 == 0):
                print("interation:", i)
                print("output: ", yp, y)
                print("Error", n.e_total)
                print()
        
    elif(sys.argv[1]=='and'):
        x=[[0, 0], [0, 1], [1, 0], [1, 1]]
        y=[0.0, 0.0, 0.0, 1.0]
        y = np.asarray(y)
        num_neurons = [1]
        n = NeuralNetwork(1, num_neurons, 2, 1, 0, 0.1)

        for e in range(2001):
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
        x=[[0, 0], [0, 1], [1, 0], [1, 1]]
        y=[0.0, 1.0, 1.0, 0.0]
        y = np.asarray(y)
        num_neurons = [2, 1]
        n = NeuralNetwork(2, num_neurons, 2, 1, 0, 0.1)

        for e in range(50001):
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