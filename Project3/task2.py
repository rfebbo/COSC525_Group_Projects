import numpy as np
import tensorflow as tf
import os
from PIL import Image

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from read_data import read_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if(len(os.sys.argv) > 1):
    if os.sys.argv[1] == 'gpu':
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    elif os.sys.argv[1] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES']="" 
else:
    os.environ['CUDA_VISIBLE_DEVICES']=""


def build_network_2(n_output):
    model=Sequential()

    # Add convolutional layers, flatten, and fully connected layer
    model.add(layers.Conv2D(40,5,input_shape=(32,32,1),activation='relu', padding='valid'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid',data_format=None))
    model.add(layers.Flatten())
    model.add(layers.Dense(100,activation='relu'))
    model.add(layers.Dense(n_output,activation='softmax'))

    return model


def main():
    dataset = read_data()

    sgd = optimizers.SGD(lr=0.01)
    loss = tf.keras.losses.CategoricalCrossentropy()

    # create race class
    model = build_network_2(7)
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    history=model.fit(dataset['train_norm'],dataset['race_t_labels'],validation_data=(dataset['val_norm'],dataset['race_v_labels']),batch_size=32,epochs=100, verbose=True)


if __name__=="__main__":
    main()