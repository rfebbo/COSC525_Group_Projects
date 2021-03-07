import NeuralNetwork as NN
from parameters import generateExample1
import numpy as np

#print needed values.
np.set_printoptions(precision=5)

n = NN.NeuralNetwork([5, 5], 0, 100)
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

input = np.reshape(input, (1,1,5,5))=

print('1st convolutional layer, 1st kernel weights:')
print(n.layers[0].weights)
print('1st convolutional layer, 1st kernel biases:')
print(n.layers[0].bias)

print('1st FC layer, weights:')
print(n.layers[2].weights)
print('1st FC layer, biases:')
print(n.layers[2].bias)

n.train(input, output)
print("\nTraining...\nfirst layer output")
print(np.squeeze(n.out[0]))
print("second layer output")
print(np.squeeze(n.out[1]))
print("final output: ", n.out[2])
print("desired output: ", output)

print(f"loss: {n.e_total}")


print('\n1st convolutional layer, 1st kernel weights:')
print(n.layers[0].weights)
print('1st convolutional layer, 1st kernel biases:')
print(n.layers[0].bias)

print('\n1 iter done\n\n1st FC layer, weights:')
n.layers[2].update_weights()
print(n.layers[2].weights)
print('1st FC layer, biases:')
print(n.layers[2].bias)
