import NeuralNetwork as NN
from parameters import generateExample3
import numpy as np

def print_nn_output(NN, input):
    NN.predict(input)
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

def run_example3(verbose):
    if(verbose):
        print("***Running COSC 525 Project Code***")

    #print needed values.
    np.set_printoptions(precision=5)

    n = NN.NeuralNetwork((1,8,8), 'MSE', 100)
    l1k1,l1k2,l1b1,l1b2,l2,input,output = generateExample3()

    #input
    input = input.reshape(1,1,8,8)

    # l1 weights
    l1k = np.append(l1k1, l1k2, axis=0)
    l1k = l1k.reshape((2,1,3,3))
    l1_weights = [l1k,np.asarray([l1b1,l1b2])]

    n.addLayer("ConvolutionLayer", numKernels = 2, kernelSize = (3,3), activation = 'sigmoid', weights=l1_weights, name='conv3')
    n.addLayer("MaxPoolingLayer", kernelSize=2, name='maxpool2')
    n.addLayer("FlattenLayer", name='flatten')
    n.addLayer("FullyConnected", numOfNeurons=1, activation='sigmoid', weights=l2, name='fullyconnected')



    if(verbose):
        print_nn_output(n,input)
        print("\nTraining...")

    n.train(input, output)

    if(verbose):
        print_nn_output(n,input)
        print(f"loss: {n.e_total}")
        
    l1k = np.squeeze(n.layers[0].weights)
    l1b = np.squeeze(n.layers[0].bias)
    l4 = np.squeeze(n.layers[3].weights)
    l4b = np.squeeze(n.layers[3].bias)

    return l1k, l1b, l4, l4b

    
if __name__=="__main__":
    run_example3(True)