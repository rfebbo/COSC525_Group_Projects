from mymath import convolve_2d
from NeuralNetwork import NeuralNetwork
import numpy as np


x = [  [[2,2,2,2,0],
        [0,2,1,1,2],
        [2,2,2,1,0],
        [2,2,1,1,1],
        [2,1,2,2,1]],

       [[2,2,0,2,1],
        [1,2,2,2,1],
        [2,2,2,1,0],
        [1,2,2,1,0],
        [0,0,1,1,1]],

       [[0,0,1,2,2],
        [0,1,0,1,1],
        [0,0,1,0,1],
        [2,2,0,1,2],
        [0,1,1,0,2]]]

f =  [[[[ 1, 0,-1],
        [-1, 1,-1],
        [ 0, 1, 0]],
       [[ 0, 1, 1],
        [ 1,-1,-1],
        [ 0, 1, 0]],
       [[ 0, 0, 0],
        [ 0, 0,-1],
        [ 1, 0, 0]]],

       [[[-1,-1,-1],
        [-1, 0, 1],
        [-1, 1,-1]],
       [[ 0,-1, 0],
        [-1, 1, 1],
        [-1, 0, 1]],
       [[ 1,-1,-1],
        [-1,-1, 1],
        [ 1, 0, 1]]]]


x = np.asarray(x).reshape(1,3,5,5)
f = np.asarray(f).reshape(2,3,3,3)
y = convolve_2d(x, f, [1, 0], 1, 1)
print('a)', y)


print('b)', np.fmax(0,y))

#print needed values.
np.set_printoptions(precision=5)

n = NeuralNetwork((3,5,5), 0, 100)
n.addLayer('ConvolutionLayer', kernelSize=(3,3), numKernels=2, stride=1, padding=1, weights=[f,[1,0]], name='conv3', activation=2)
n.addLayer('MaxPoolingLayer', kernelSize=2, stride=1, name='maxpool2')
n.addLayer('FlattenLayer', name='flatten')
n.addLayer('FullyConnected', numOfNeurons=2, name='fully connected 2', activation=1)

n.calculate(x)

for l in n.layers:
        print('layer: ')
        print(l.name)
        print('\toutput:\n ')
        print(l.out)
        

#problem 3
print('problem 3')

#input
x = [1,0,1,0,1,0,1,0,1]
x = np.asarray(x).reshape(1,3,3)
y = np.asarray([1,0])

#layer 1 kernel and bias
l1k1 = [0.9, 0.1, 0.1, 0.9]
l1k1 = np.asarray(l1k1).reshape(1,1,2,2)
l1b1 = [0.1]

#FC layer weights
l3w = np.arange(0.1, 0.9, 0.1).reshape(2,4)
l3b = [[0.2, 0.2]]
l4w = np.arange(0.1, 0.5, 0.1).reshape(2,2)
l4b = [[0.3, 0.3]]

l3w = np.append(l3w,np.transpose(l3b),axis=1)
l4w = np.append(l4w,np.transpose(l4b),axis=1)
print(l3w)
print(l3w.shape)
# exit()

n = NeuralNetwork((1,3,3), "SCCE", 0.5)
n.addLayer('ConvolutionLayer', kernelSize=(2,2), numKernels=1, stride=1, padding=0, weights=[l1k1,l1b1], name='conv2', activation=2)
n.addLayer('FlattenLayer', name='flatten')
n.addLayer('FullyConnected', numOfNeurons=2, name='fully connected 2', activation=1, weights=l3w)
n.addLayer('FullyConnected', numOfNeurons=2, name='fully connected 2', activation=1, weights=l4w)

n.calculate(x) #fill the .out parts with stuff so we can print it

for l in n.layers:
        print('\nlayer: ')
        print(l.name)
        print('\toutput: ')
        print(l.out)
        if l.hasWeights():
                print('\tweights: ')
                print(l.weights)

print('\nTraining...')
n.train(x,y)

for l in n.layers:
        print('\nlayer: ', l.name)
        print('\toutput:\n ', l.out)
        if l.hasWeights():
                print('\tweights: ')
                print(l.weights)