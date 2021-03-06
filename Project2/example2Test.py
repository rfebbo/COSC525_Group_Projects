import NeuralNetwork as NN
from parameters import generateExample2
import numpy as np

# n = NN.NeuralNetwork([5, 5], 1, 100)
# l1k1, l1b1, l2, l2b, input, output = generateExample1()
# input1 = [[input]]
# l1k1=l1k1.reshape(1,1,3,3)
# weights = ([l1k1,np.array([l1b1[0]])])
# fullyconnectedweights  = []
# for i in range(len(l2[0])):
#     fullyconnectedweights.append(l2[0][i])
# fullyconnectedweights.append(l2b[0])

n = NN.NeuralNetwork([7,7], 1, 100)
l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, output = generateExample2()
# l1 weights
l1k = np.append(l1k1, l1k2, axis=0)
l1k = l1k.reshape((2,1,3,3))
l1_weights = [l1k,np.asarray([l1b1,l1b2])]
# print(l1k.shape)    
# print(l1k)

# l2 weights
l2c = np.append(l2c1, l2c2, axis=0)
l2c = l2c.reshape((1,2,3,3))
l2_weights = [l2c,np.asarray([l2b])]

# fully connnected weights
l3_weights = []
for i in range(len(l3[0])):
    l3_weights.append(l3[0][i])
l3_weights.append(l3b[0])

input = input.reshape(1,1,7,7)
n.addLayer("ConvolutionLayer", numKernels = 2, kernelSize = (3,3), activation = 1, inputDim = (1, 1, 7, 7), weights=l1_weights)
n.addLayer("ConvolutionLayer", numKernels = 1, kernelSize = (3,3), activation = 1, inputDim = (1, 2, 7, 7), weights=l2_weights)
n.addLayer("FlattenLayer", inputDim=[1,3,3])
n.addLayer("FullyConnected", numOfNeurons=1, activation=1, input_num=9, weights=[l3_weights])
out = n.layers[0].calculate(np.asarray(input))
print("first layer output\n", out)
out = n.layers[1].calculate(out)
print("second layer output\n", out)
out = n.layers[2].calculate(out)
print("Third layer output\n", out)
out = n.layers[3].calculate(out)
print("final output: ", out)