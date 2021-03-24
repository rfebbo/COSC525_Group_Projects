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


def build_network_1(n_output):
    model=Sequential()

    # Add convolutional layers, flatten, and fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(1024,activation='tanh'))
    model.add(layers.Dense(512,activation='sigmoid'))
    model.add(layers.Dense(100,activation='relu'))
    model.add(layers.Dense(n_output,activation='softmax'))

    return model


def main():
    dataset = read_data()

    sgd = optimizers.SGD(lr=0.05)
    loss = tf.keras.losses.BinaryCrossentropy()

    # create race class
    # model = build_network(7)
    # model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    # history=model.fit(dataset['train_norm'],dataset['race_t_labels'],validation_data=(dataset['val_norm'],dataset['race_v_labels']),batch_size=4096,epochs=100, verbose=True)

    # create gender class
    model = build_network_1(2)
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    history=model.fit(dataset['train_norm'],dataset['gender_t_labels'],validation_data=(dataset['val_norm'],dataset['gender_v_labels']),batch_size=64,epochs=100, verbose=True)


if __name__=="__main__":
    main()