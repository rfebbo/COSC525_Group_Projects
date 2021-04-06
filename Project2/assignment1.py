from mymath import convolve_2d
from NeuralNetwork import NeuralNetwork
import numpy as np


x =[[1, 0, 0, 2],
    [0, 1, 0, 1],
    [0, 0, 1, 2],
    [0, 0, 0, 1]]

f = [[0, 0], [1, 1]]

x = np.asarray(x).reshape(1,1,4,4)
f = np.asarray(f).reshape(1,1,2,2)
y = convolve_2d(x, f, [0, 0], 1, 0)
print('a)', y)

x =[[[1, 0, 0, 2],
    [0, 1, 0, 1],
    [0, 0, 1, 2],
    [0, 0, 0, 1]],
    [[1, 0, 0, 2],
    [0, 1, 0, 1],
    [0, 0, 1, 2],
    [0, 0, 0, 1]]]

f = [[[0, 0], [1, 1]],[[0, 0], [1, 1]]]

x = np.asarray(x).reshape(1,2,4,4)
f = np.asarray(f).reshape(1,2,2,2)
y = convolve_2d(x, f, [0, 0], 1, 0)
print('\nb)', y)

x =[[1, 0, 0, 2],
    [0, 1, 0, 1],
    [0, 0, 1, 2],
    [0, 0, 0, 1]]

f = [[[0, 0], [1, 1]],[[0, 0], [1, 1]]]

x = np.asarray(x).reshape(1,1,4,4)
f = np.asarray(f).reshape(2,1,2,2)
y = convolve_2d(x, f, [0, 0], 1, 0)
print('\nc)', y)

n = NeuralNetwork((21,21,3), 0, 100)
n.addLayer('ConvolutionLayer', kernelSize=5, numKernels=6, stride=2, padding=0)
n.addLayer('MaxPoolingLayer', kernelSize=3, stride=2)
n.addLayer('FlattenLayer')
n.addLayer('FullyConnected',numOfNeurons=10)


