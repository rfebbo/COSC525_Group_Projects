import NeuralNetwork as NN
from parameters import generateExample2
import numpy as np

def print_nn_wandb(NN, input):
    NN.predict(input)
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

def run_example2(verbose):
    if(verbose):
        print("***Running COSC 525 Project Code***")
    #print needed values.
    np.set_printoptions(precision=5)

    n = NN.NeuralNetwork((1,7,7), 'MSE', 100)
    l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,input, output = generateExample2()
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


    input = input.reshape(1, 1,7,7)
    n.addLayer("ConvolutionLayer", numKernels = 2, kernelSize = (3,3), activation = 'sigmoid', weights=l1_weights, name="conv3_1")
    n.addLayer("ConvolutionLayer", numKernels = 1, kernelSize = (3,3), activation = 'sigmoid', weights=l2_weights, name='conv3_2')
    n.addLayer("FlattenLayer", name="Flatten")
    n.addLayer("FullyConnected", numOfNeurons=1, activation='sigmoid', weights=l3, name="FullyConnected")

    if (verbose):
        print_nn_wandb(n, input)
        print("\nTraining...\n")

    n.train(input, output)

    if(verbose):
        print_nn_wandb(n, input)
        print(f"loss: {n.e_total}")

        
    n.layers[3].update_weights() #update the layer copy of the weights(neurons have newer weights) for logging

    l1k = np.squeeze(n.layers[0].weights)
    l1b = np.squeeze(n.layers[0].bias)
    l2k = np.squeeze(n.layers[1].weights)
    l2b = np.squeeze(n.layers[1].bias)
    l4 = np.squeeze(n.layers[3].weights)
    l4b = np.squeeze(n.layers[3].bias)

    return l1k, l1b, l2k, l2b, l4, l4b
    
if __name__=="__main__":
    run_example2(True)