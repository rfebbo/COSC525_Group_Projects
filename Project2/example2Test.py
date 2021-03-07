import NeuralNetwork as NN
from parameters import generateExample2
import numpy as np
#print needed values.
np.set_printoptions(precision=5)

# n = NN.NeuralNetwork([5, 5], 1, 100)
# l1k1, l1b1, l2, l2b, input, output = generateExample1()
# input1 = [[input]]
# l1k1=l1k1.reshape(1,1,3,3)
# weights = ([l1k1,np.array([l1b1[0]])])
# fullyconnectedweights  = []
# for i in range(len(l2[0])):
#     fullyconnectedweights.append(l2[0][i])
# fullyconnectedweights.append(l2b[0])

n = NN.NeuralNetwork([7,7], 0, 100)
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

input = input.reshape(1, 1,7,7)
n.addLayer("ConvolutionLayer", numKernels = 2, kernelSize = (3,3), activation = 1, inputDim = (1, 1, 7, 7), weights=l1_weights, name="conv3_1")
n.addLayer("ConvolutionLayer", numKernels = 1, kernelSize = (3,3), activation = 1, weights=l2_weights, name='conv3_2')
n.addLayer("FlattenLayer", name="Flatten")
n.addLayer("FullyConnected", numOfNeurons=1, activation=1, weights=[l3_weights], name="FullyConnected")

def print_nn_wandb(NN, input):
    n.predict(input)
    print('\n1st convolutional layer, 1st kernel weights:')
    print(NN.layers[0].weights)
    print('1st convolutional layer, 1st kernel biases:')
    print(NN.layers[0].bias)
    print("first layer output")
    print(np.squeeze(NN.out[0]))

    print('\n2nd convolutional layer, 1st kernel weights:')
    print(NN.layers[1].weights)
    print('2nd convolutional layer, 1st kernel biases:')
    print(NN.layers[1].bias)
    print("second layer output")
    print(np.squeeze(NN.out[1]))
    
    print("\nthird layer output")
    print(np.squeeze(NN.out[2]))

    NN.layers[3].update_weights()
    print('\n1st FC layer, weights:')
    print(NN.layers[3].weights)
    print('1st FC layer, biases:')
    print(NN.layers[3].bias)
    
    print("final output: ", NN.out[3])


print_nn_wandb(n, input)


print("\nTraining...\n")
n.train(input, output)

print_nn_wandb(n, input)

print(f"loss: {n.e_total}")

# # calculate total error
# n.e_total = n.calculateloss(n.out[-1], output)

# # calculate d_error for last layer
# d_error = n.lossderiv(n.out[-1], output)

# # calculate delta for last layer
# wdelta = []
# for i, nue in enumerate(n.layers[-1].neurons):
#     nue.calcpartialderivative(d_error[i])
#     wdelta_i = (nue.weights * nue.delta)
#     wdelta.append(wdelta_i)
    
#     # print("neuron: ", i, " weights: ", n.weights, "bias: ", n.bias)
#     nue.updateweight()

# wdelta = np.sum(wdelta, axis=0)
# n.layers[3].update_weights()

# # update weights using delta
# # for i, l in enumerate(reversed(n.layers)):
# #     print(i)
# #     if i == 0:
# #         continue
# #     print(l.name)
# #     wdelta = l.calcwdeltas(wdelta)

# print('\n1st FC layer, weights:')
# print(n.layers[3].weights)
# print('1st FC layer, biases:')
# print(n.layers[3].bias)


# print(n.layers[2].name)
# wdelta = n.layers[2].calcwdeltas(wdelta)
# print(n.layers[1].name)
# wdelta = n.layers[1].calcwdeltas(wdelta)
# print('2nd convolutional layer, kernel weights:')
# print(n.layers[1].weights)
# print('2nd convolutional layer, biases:')
# print(n.layers[1].bias)
# print(n.layers[0].name)
# wdelta = n.layers[0].calcwdeltas(wdelta)


# print('1st convolutional layer, kernel weights:')
# print(n.layers[0].weights)
# print('1st convolutional layer, biases:')
# print(n.layers[0].bias)
# # print_nn_wandb(n)