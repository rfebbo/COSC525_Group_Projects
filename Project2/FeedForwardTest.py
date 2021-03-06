import NeuralNetwork as NN
from parameters import generateExample1
import numpy as np

n = NN.NeuralNetwork([5, 5], 1, 100)
l1k1, l1b1, l2, l2b, input, output = generateExample1()
input1 = [[input]]
l1k1=l1k1.reshape(1,1,3,3)
weights = ([l1k1,np.array([l1b1[0]])])
fullyconnectedweights  = []
for i in range(len(l2[0])):
    fullyconnectedweights.append(l2[0][i])
fullyconnectedweights.append(l2b[0])

n.addLayer("ConvolutionLayer", numKernels = 1, kernelSize = (3,3), activation = 1, inputDim = (1, 1, 5, 5), weights=weights)
n.addLayer("FlattenLayer", inputDim=[1,3,3])
n.addLayer("FullyConnected", numOfNeurons=1, activation=1, input_num=9, weights=[fullyconnectedweights])
# print(n.layers[0].weights)
out = n.layers[0].calculate(np.asarray(input1))
print("first layer output\n", out)
out = n.layers[1].calculate(out)
print("second layer output\n", out)
out = n.layers[2].calculate(out)
print("final output: ", out)