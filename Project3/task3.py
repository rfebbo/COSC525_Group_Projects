import numpy as np
import tensorflow as tf
import os
from PIL import Image
import pandas as pd

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from read_data import read_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if(len(os.sys.argv) > 1):
    if os.sys.argv[1] == 'gpu':
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    elif os.sys.argv[1] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES']="" 
else:
    os.environ['CUDA_VISIBLE_DEVICES']=""


from model_runner import run_all_models


def build_network_3(n_output):
    model=Sequential()

    # Add convolutional layers, flatten, and fully connected layer
    model.add(layers.Conv2D(20,3,input_shape=(32,32,1),activation='relu', padding='valid'))
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='valid',data_format=None))
    model.add(layers.Conv2D(20,3,activation='relu', padding='valid'))
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='valid',data_format=None))
    model.add(layers.Flatten())
    model.add(layers.Dense(100,activation='relu'))
    model.add(layers.Dense(n_output,activation='softmax'))

    return model


def test_network_3():
    d = read_data()

    lrs = [0.05]        
    #lrs = [0.01,s 0.05, 0.1]
    momentum = 0.9
    batch_size =  128
    epochs = 100

    for lr in lrs:
        run_all_models(build_network_3, 'task_3', d, lr, momentum, batch_size, epochs)

if __name__=="__main__":
    test_network_3()