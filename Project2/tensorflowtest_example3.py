import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
if os.sys.argv[1] == 'gpu':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
elif os.sys.argv[1] == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES']="" 

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import generateExample3
from mymath import convolve_2d

def print_model_info(model,input_img):
    extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])

    features = extractor(input_img)
    layer1_out = np.expand_dims(features[0][0],axis=0)
    layer1_out = np.squeeze(layer1_out).reshape(2,6,6)
    layer2_out = np.expand_dims(features[1][0],axis=0)
    layer2_out = tf.squeeze(layer2_out)
    layer3_out = np.expand_dims(features[2][0],axis=0)
    layer3_out = tf.squeeze(layer3_out)
    layer4_out = np.expand_dims(features[3][0],axis=0)
    layer4_out = tf.squeeze(layer4_out)
    print('model output:')
    print(model.predict(img))

    print('\n1st convolutional layer, 1st kernel weights:')
    print(np.squeeze(model.get_weights()[0][:,:,0,0]))
    print('1st convolutional layer, 1st kernel bias:')
    print(np.squeeze(model.get_weights()[1][0]))
    print('1st convolutional layer, 2nd kernel weights:')
    print(np.squeeze(model.get_weights()[0][:,:,0,1]))
    print('1st convolutional layer, 2nd kernel bias:')
    print(np.squeeze(model.get_weights()[1][1]))
    print('1st convolutional layer, output:')
    print(np.asarray(layer1_out))
    
    print("\nMax Pooling layer, output")
    print(np.asarray(layer2_out))

    print('\nflatten layer, output:')
    print(np.asarray(layer3_out))
    
    print('\nFully connnected layer weights')
    print(np.squeeze(model.get_weights()[2]))
    print('Fully connected layer bias')
    print(np.squeeze(model.get_weights()[3]))
    print('fully connected layer, output:')
    print(np.asarray(layer4_out))

def run_tf_example3(verbose):
    #print needed values.
    np.set_printoptions(precision=5)


    #Create a feed forward network
    model=Sequential()
    #model.add(layers.Conv2D(2,3,input_shape=(7,7,1),activation='sigmoid')) 
    model.add(layers.Conv2D(2,3,input_shape=(8,8,1),activation='sigmoid'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid',data_format=None))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    l1k1,l1k2,l1b1,l1b2,l2,l2b,input,output = generateExample3()

    #setting weights and bias of first layer.
    l1k1=l1k1.reshape(3,3,1,1)
    l1k2=l1k2.reshape(3,3,1,1)

    w1=np.concatenate((l1k1,l1k2),axis=3)
    model.layers[0].set_weights([w1,np.array([l1b1[0],l1b2[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)

    #setting weights and bias of fully connected layer.
    model.layers[3].set_weights([np.transpose(l2),l2b])

    #Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
    img=np.expand_dims(input,axis=(0,3))
    
    if(verbose):
        print_model_info(model,img)
        print('\ntraining...')

    sgd = optimizers.SGD(lr=100)
    model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
    history=model.fit(img,output,batch_size=1,epochs=1,verbose=verbose)

    if(verbose):
        print_model_info(model,img)
        print('loss: ', history.history['loss'])

    l1k = np.squeeze([model.get_weights()[0][:,:,0,0],model.get_weights()[0][:,:,0,1]]).reshape(2,3,3)
    l1b = np.squeeze(model.get_weights()[1][:])

    l4 = np.squeeze(model.get_weights()[2])
    l4b = np.squeeze(model.get_weights()[3])

    return l1k, l1b, l4, l4b


