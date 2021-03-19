import NeuralNetwork as NN
from parameters import generateExample1
import numpy as np

def print_nn_info(NN,input):

    NN.predict(input)
    print('1st convolutional layer, 1st kernel weights:')
    print(np.squeeze(NN.layers[0].weights))
    print('1st convolutional layer, 1st kernel biases:')
    print(NN.layers[0].bias)
    print("1st convolutional layer, output:")
    print(np.squeeze(NN.out[0]))

    print("\nflatten layer, output:")
    print(np.squeeze(NN.out[1]))

    print('\nfully connected layer weights:')
    NN.layers[2].update_weights()
    print(np.squeeze(NN.layers[2].weights))
    print('fully connected layer bias:')
    print(NN.layers[2].bias)
    
    print("final output: ", NN.out[2])

def run_example1(verbose):
    if(verbose):
        print("***Running COSC 525 Project Code***")
    #print needed values.
    np.set_printoptions(precision=5)

    n = NN.NeuralNetwork((1,5, 5), loss="MSE", lr=100)
    l1k1, l1b1, l2, input, output = generateExample1()

    l1k1=l1k1.reshape(1,1,3,3)
    weights = ([l1k1,np.array([l1b1[0]])])
    

    n.addLayer("ConvolutionLayer", numKernels = 1, kernelSize = (3,3), activation = 'sigmoid', weights=weights, name='conv1')
    n.addLayer("FlattenLayer")
    n.addLayer("FullyConnected", numOfNeurons=1, activation='sigmoid', weights=l2, name='fully connected 1')

    input = np.reshape(input, (1,1,5,5))
    if(verbose):
        print_nn_info(n, input)
        print("\nTraining...\n")

    n.train(input, output)

    if(verbose):
        print_nn_info(n, input)
        print(f"loss: {n.e_total}")

    l1k1 = np.squeeze(n.layers[0].weights)
    l1b1 = np.squeeze(n.layers[0].bias)
    l2 = np.squeeze(n.layers[2].weights)
    l2b = np.squeeze(n.layers[2].bias)

    return l1k1, l1b1, l2, l2b

    
if __name__=="__main__":
    run_example1(True)

