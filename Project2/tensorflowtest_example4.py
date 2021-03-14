import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
if(len(os.sys.argv) > 1):
    if os.sys.argv[1] == 'gpu':
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    elif os.sys.argv[1] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=""
else:
    os.environ['CUDA_VISIBLE_DEVICES']=""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import generateExample4
from mymath import convolve_2d

def print_model_info(model,input_img, output, loss):
    extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])

    features = extractor(input_img)
    layer1_out = np.expand_dims(features[0][0],axis=0)
    layer1_out = np.squeeze(layer1_out)
    layer2_out = np.expand_dims(features[1][0],axis=0)
    layer2_out = tf.squeeze(layer2_out)
    layer3_out = np.expand_dims(features[2][0],axis=0)
    layer3_out = tf.squeeze(layer3_out)
    layer4_out = np.expand_dims(features[3][0],axis=0)
    layer4_out = tf.squeeze(layer4_out)

    print('\n1st convolutional layer, 1st kernel weights:')
    print(np.squeeze(model.get_weights()[0][:,:,0,0]))
    print('1st convolutional layer, 1st kernel bias:')
    print(np.squeeze(model.get_weights()[1][0]))
    print('1st convolutional layer, output:')
    print(np.asarray(layer1_out))
    
    print('\nflatten layer, output:')
    print(np.asarray(layer2_out))

    print('\nFully connected layer 1 weights')
    print(np.squeeze(model.get_weights()[2]))
    print('Fully connected layer 1 bias')
    print(np.squeeze(model.get_weights()[3]))
    print('fully connected layer 1, output:')
    print(np.asarray(layer3_out))
    
    print('\nFully connected layer 2 weights')
    print(np.squeeze(model.get_weights()[4]))
    print('Fully connected layer 2 bias')
    print(np.squeeze(model.get_weights()[5]))
    print('fully connected layer 2, output:')
    print(np.asarray(layer4_out))

    print('Target Output: ', output)
    
    prediction = model.predict(input_img)
    print('loss: ', loss(output, prediction).numpy())

def run_tf_example4(verbose):
    if(verbose):
        print("\n***RUNNING TENSORFLOW***")
    #print needed values.
    np.set_printoptions(precision=5)


    #Create a feed forward network
    model=Sequential()
    #model.add(layers.Conv2D(2,3,input_shape=(7,7,1),activation='sigmoid')) 
    model.add(layers.Conv2D(1,2,input_shape=(3,3,1),activation='sigmoid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2,activation='sigmoid'))
    model.add(layers.Dense(2,activation='sigmoid'))

    l1,l3,l4,input,output = generateExample4()

    #TF dimensions are W,H,N,C
    l1[0]=l1[0].reshape(2,2,1,1)

    model.layers[0].set_weights(l1) #Shape of weight matrix is (w,h,input_channels,kernels)

    #setting weights and bias of fully connected layers.
    model.layers[2].set_weights(l3)
    model.layers[3].set_weights(l4)

    #Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
    input=input.reshape(1,3,3,1)

        
    sgd = optimizers.SGD(lr=0.5)
    # loss = tf.keras.losses.CategoricalCrossentropy()
    loss = tf.keras.losses.MeanSquaredError()

    if(verbose):
        print_model_info(model,input, output, loss)
        print('\ntraining...')
    

    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    history=model.fit(input,output,batch_size=1,epochs=1,verbose=verbose)

    if(verbose):
        print_model_info(model,input, output, loss)
        print('loss: ', history.history['loss']) #why do the losses not match?  

    l1k = np.squeeze([model.get_weights()[0][:,:,0,0],model.get_weights()[0][:,:,0,0]])
    l1b = np.squeeze(model.get_weights()[1][:])

    l3 = np.squeeze(model.get_weights()[2]) 
    l3b = np.squeeze(model.get_weights()[3])

    l4 = np.squeeze(model.get_weights()[4]) 
    l4b = np.squeeze(model.get_weights()[5])

    return l1k, l1b, l3, l3b, l4, l4b


if __name__=="__main__":
    run_tf_example4(True)