import numpy as np

#Generate data and weights for "example1"
def generateExample1():
    #Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(10)

    #First hidden layer, two kernels
    l1k1=np.random.rand(3,3)
    l1b1=np.random.rand(1)

    #output layer, fully connected
    l2 = np.random.rand(9,1)
    l2b=np.random.rand(1)
    l2 = [l2,l2b]

    #input and output
    input=np.random.rand(5,5)
    output=np.random.rand(1)
    return l1k1,l1b1,l2,input,output

#Generate data and weights for "example2"
def generateExample2():
    #Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(10)

    #First hidden layer, two kernels
    l1k1=np.random.rand(3,3)
    l1k2=np.random.rand(3,3)
    l1b1=np.random.rand(1)
    l1b2=np.random.rand(1)

    #second hidden layer, one kernel, two channels
    l2c1=np.random.rand(3,3)
    l2c2=np.random.rand(3,3)
    l2b=np.random.rand(1)

    #output layer, fully connected
    l3=np.random.rand(9,1)
    l3b=np.random.rand(1)
    l3 = [l3,l3b]

    #input and output
    input=np.random.rand(7,7)
    output=np.random.rand(1)

    return l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,input,output

#Generate data and weights for "example3"
def generateExample3():
    #Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(12)

    #First hidden layer, two kernels
    l1k1=np.random.rand(3,3)
    l1k2=np.random.rand(3,3)
    l1b1=np.random.rand(1)
    l1b2=np.random.rand(1)


    #output layer, fully connected
    l4=np.random.rand(18,1)
    l4b=np.random.rand(1)
    l4 = [l4,l4b]

    #input and output
    input=np.random.rand(8,8)
    output=np.random.rand(1)

    return l1k1,l1k2,l1b1,l1b2,l4,input,output


#Generate data and weights for "example3"
def generateExample4():
    #input
    input = np.asarray([1,0,1,0,1,0,1,0,1]).reshape(1,3,3)
    output = np.asarray([[1,0]])

    #layer 1 kernel and bias
    l1k1 = np.asarray([0.9, 0.1, 0.1, 0.9]).reshape(1,1,2,2)
    l1b1 = np.asarray([0.1])
    l1 = [l1k1, l1b1]

    #FC layer weights
    l3w = np.transpose(np.arange(0.1, 0.9, 0.1).reshape(2,4))
    l3b = np.asarray([0.2, 0.2])
    l3 = [l3w, l3b]

    l4w = np.transpose(np.arange(0.1, 0.5, 0.1).reshape(2,2))
    l4b = np.asarray([0.3, 0.3])
    l4 = [l4w, l4b]

    return l1,l3,l4,input,output

