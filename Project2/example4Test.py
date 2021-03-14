import NeuralNetwork as NN
from parameters import generateExample4
import numpy as np

def print_nn_output(NN, input, output):
    NN.predict(input)
    print('\n1st convolutional layer, 1st kernel weights:')
    print(NN.layers[0].weights)
    print('1st convolutional layer, 1st kernel biases:')
    print(NN.layers[0].bias)
    print("first layer output")
    print(np.squeeze(NN.out[0]))

    print("\nflatten layer output")
    print(np.squeeze(NN.out[1]))
    
    NN.layers[2].update_weights()
    print('\nFully connected layer 1 weights')
    print(NN.layers[2].weights)
    print('Fully connected layer 1 bias')
    print(NN.layers[2].bias)
    print('fully connected layer 1, output:')
    print(np.squeeze(NN.out[2]))

    NN.layers[3].update_weights()
    print('\nFully connected layer 2 weights')
    print(NN.layers[3].weights)
    print('Fully connected layer 2 bias')
    print(NN.layers[3].bias)
    print('fully connected layer 1, output:')
    print(NN.out[3])

    print('Target Output: ', output)

    print("loss: ", NN.calculateloss(NN.out[3], output))
    
    # print("d_loss/d_y : ", NN.lossderiv(NN.out[3], output))

def run_example4(verbose):
    if(verbose):
        print("***Running COSC 525 Project Code***")

    #print needed values.
    np.set_printoptions(precision=5)

    n = NN.NeuralNetwork((1,8,8), 0, 100)
    l1,l3,l4,input,output = generateExample4()

    #input
    input = input.reshape(1,1,3,3)

    n = NN.NeuralNetwork((1,3,3), loss="MSE", lr=0.5)
    n.addLayer('ConvolutionLayer', kernelSize=(2,2), numKernels=1, stride=1, padding=0, weights=l1, name='conv2', activation='sigmoid')
    n.addLayer('FlattenLayer', name='flatten')
    n.addLayer('FullyConnected', numOfNeurons=2, name='fully connected 1', activation='sigmoid', weights=l3)
    n.addLayer('FullyConnected', numOfNeurons=2, name='fully connected 2', activation='sigmoid', weights=l4)


    if(verbose):
        print_nn_output(n, input, output)
        print("\nTraining...\n")

    n.train(input, output)

    if(verbose):
        print_nn_output(n, input, output)
        print(f"loss: {n.e_total}")
        
    l1k = np.squeeze(n.layers[0].weights)
    l1b = np.squeeze(n.layers[0].bias)
    l3 = np.squeeze(n.layers[2].weights)
    l3b = np.squeeze(n.layers[2].bias)
    l4 = np.squeeze(n.layers[3].weights)
    l4b = np.squeeze(n.layers[3].bias)

    return l1k, l1b, l3, l3b, l4, l4b

    
if __name__=="__main__":
    run_example4(True)