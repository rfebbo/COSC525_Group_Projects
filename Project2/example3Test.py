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
n.addLayer("MaxPoolingLayer", kernelSize=2)
n.addLayer("FlattenLayer")
n.addLayer("FullyConnected", numOfNeurons=1, activation=1, weights=[l2_weights])


def print_nn_output(NN, input):
    n.predict(input)
    print('\n1st convolutional layer, 1st kernel weights:')
    print(NN.layers[0].weights)
    print('1st convolutional layer, 1st kernel biases:')
    print(NN.layers[0].bias)
    print("first layer output")
    print(np.squeeze(NN.out[0]))

    print("MaxPool layer output")
    print(np.squeeze(NN.out[1]))
    
    print("\nthird layer output")
    print(np.squeeze(NN.out[2]))

    NN.layers[3].update_weights()
    print('\n1st FC layer, weights:')
    print(NN.layers[3].weights)
    print('1st FC layer, biases:')
    print(NN.layers[3].bias)
    
    print("final output: ", NN.out[3])

# out = n.layers[0].calculate(np.asarray(input))
# print("first layer output\n", out[0])
# print("second layer output\n", out)
# print("Third layer output\n",out)
# print("Final output: ", out)



n.calculate(input)    
# print_nn_output(n,input)

# n.train(input, output)

# print(f"loss: {n.e_total}")

# # calculate total error
n.e_total = n.calculateloss(n.out[-1], output)

# # # calculate d_error for last layer
# d_error = n.lossderiv(n.out[-1], output)

# # # calculate delta for last layer
# wdelta = []
# for i, nue in enumerate(n.layers[-1].neurons):
#     nue.calcpartialderivative(d_error[i])
#     wdelta_i = (nue.weights * nue.delta)
#     wdelta.append(wdelta_i)
    
#     # print("neuron: ", i, " weights: ", n.weights, "bias: ", n.bias)
#     nue.updateweight()

# wdelta = np.sum(wdelta, axis=0)
# n.layers[3].update_weights()

# print('\n1st FC layer, weights:')
# print(n.layers[3].weights)
# print('1st FC layer, biases:')
# print(n.layers[3].bias)

# # update weights using delta
# # for i, l in enumerate(reversed(n.layers)):
# #     print(i)
# #     if i == 0:
# #         continue
# #     print(l.name)
# #     wdelta = l.calcwdeltas(wdelta)



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