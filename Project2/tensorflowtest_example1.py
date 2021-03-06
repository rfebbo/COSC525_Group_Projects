

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
if os.sys.argv[1] == 'gpu':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
elif os.sys.argv[1] == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES']="" 

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import generateExample1
from mymath import convolve_2d



#Create a feed forward network
model=Sequential()

# Add convolutional layers, flatten, and fully connected layer
model.add(layers.Conv2D(1,3,input_shape=(5,5,1),activation='sigmoid', name='conv1'))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation='sigmoid'))

# Call weight/data generating function
l1k1, l1b1, l2, l2b, input, output = generateExample1()

#Set weights to desired values 

#setting weights and bias of first layer.
l1k1=l1k1.reshape(3,3,1,1)

model.layers[0].set_weights([l1k1,np.array([l1b1[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)


#setting weights and bias of fully connected layer.
model.layers[2].set_weights([np.transpose(l2),l2b])

#Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
img=np.expand_dims(input,axis=(0,3))


def print_model_info(model,input_img):
    extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])

    features = extractor(input_img)
    layer1_out = np.expand_dims(features[0][0],axis=0)
    layer1_out = tf.squeeze(layer1_out)
    layer2_out = np.expand_dims(features[1][0],axis=0)
    layer2_out = tf.squeeze(layer2_out)
    layer3_out = np.expand_dims(features[2][0],axis=0)
    layer3_out = tf.squeeze(layer3_out)
    print('model output:')
    print(model.predict(img))

    print('1st convolutional layer, 1st kernel weights:')
    print(np.squeeze(model.get_weights()[0][:,:,0,0]))
    print('1st convolutional layer, 1st kernel bias:')
    print(np.squeeze(model.get_weights()[1][0]))
    print('1st convolutional layer, output:')
    print(np.asarray(layer1_out))

    print('\nflatten layer, output:')
    print(np.asarray(layer2_out))

    print('\nfully connected layer weights:')
    print(np.squeeze(model.get_weights()[2]))
    print('fully connected layer bias:')
    print(np.squeeze(model.get_weights()[3]))
    print('fully connected layer, output:')
    print(np.asarray(layer3_out))


#print needed values.
np.set_printoptions(precision=5)

img_test = np.reshape(img, (1,1,5,5))
l1k1_test = np.reshape(img, (1,1,3,3))
y = convolve_2d(img_test,l1k1_test,l1b1,1,0)
y = 1 / (1 + np.exp(-y))
print("our output:")
print(y)

print_model_info(model, img)

sgd = optimizers.SGD(lr=100)
print('\ntraining...')
model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
history=model.fit(img,output,batch_size=1,epochs=1)

print()
print_model_info(model, img)


