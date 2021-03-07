import NeuralNetwork as NN
from parameters import generateExample3
import numpy as np

#print needed values.
np.set_printoptions(precision=5)

n = NN.NeuralNetwork([8,8], 1, 100)
l1k1,l1k2,l1b1,l1b2,l2,l2b,input,output = generateExample3()

#input
input = input.reshape(1,1,8,8)

# l1 weights
l1k = np.append(l1k1, l1k2, axis=0)
l1k = l1k.reshape((2,1,3,3))
l1_weights = [l1k,np.asarray([l1b1,l1b2])]
# print(l1k.shape)    
# print(l1k)

#l2 weights
l2_weights = []
for i in range(len(l2[0])):
    l2_weights.append(l2[0][i])
l2_weights.append(l2b[0])
print(l2_weights)
n.addLayer("ConvolutionLayer", numKernels = 2, kernelSize = (3,3), activation = 1, inputDim = (1, 1, 8, 8), weights=l1_weights)
n.addLayer("MaxPoolingLayer", kernelSize=2, inputDim=(2,6,6))
n.addLayer("FlattenLayer", inputDim = [2, 2, 2])
n.addLayer("FullyConnected", numOfNeurons=1, activation=1, input_num=18, weights=[l2_weights])


def print_nn_output(NN):
    print("first layer output")
    print(np.squeeze(NN.out[0]))
    print("second layer output")
    print(np.squeeze(NN.out[1]))
    print("third layer output")
    print(np.squeeze(NN.out[2]))
    print("final output: ", NN.out[3])

out = n.layers[0].calculate(np.asarray(input))
print("first layer output\n", out[0])
out = n.layers[1].calculate(out[0])
print("second layer output\n", out)
out = n.layers[2].calculate(out)
print("Third layer output\n",out)
out = n.layers[3].calculate(out)
print("Final output: ", out)
