import numpy as np
import tensorflow as tf
import os
from PIL import Image
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras import Model

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


def run_model_4(builder, name, n_classes, lr, momentum, x, y, val_data, batch_size, epochs):

    print('running ' + name)
    sgd = optimizers.SGD(lr=lr,momentum=momentum)
    loss = tf.keras.losses.CategoricalCrossentropy()

    model = builder(n_classes)
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    history=model.fit(x,y,validation_data=val_data,batch_size=batch_size,epochs=epochs, verbose=True)

    pd.DataFrame.from_dict(history.history,orient='index').to_csv('./saved_runs/'+name + '(lr_' + str(lr) + ')(batch_' + str(batch_size) + ')(epoch_' + str(epochs) + ')' + '.csv')


def build_network_4(n_output):
    # model=Sequential()

    # Add convolutional layers, flatten, and fully connected layer
    inputs = Input(shape = (32,32,1))
    x = layers.Conv2D(20,3,activation='relu', padding='valid')(inputs)
    x = layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='valid',data_format=None)(x)
    x = layers.Conv2D(20,3,activation='relu', padding='valid')(x)
    x = layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='valid',data_format=None)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100,activation='relu')(x)
    outputs_1 = layers.Dense(n_output[0],activation='softmax', name='race')(x)
    outputs_2 = layers.Dense(n_output[1],activation='softmax' , name='age')(x)

    model = Model(inputs=inputs,outputs=[outputs_1,outputs_2])
    return model


def test_network_4():
    d = read_data()

    # lrs = [0.05]
    lrs = [0.01, 0.1]
    momentum = 0.9
    batch_size =  128
    epochs = 100

    y = {'race' : d['race_t_labels'], 'age' : d['age_t_labels']}
    n_classes = [len(d['race_classes']), len(d['age_classes'])]
    val_data = (d['val'],[d['race_v_labels'], d['age_v_labels']])

    for lr in lrs:
        run_model_4(build_network_4, 'task_4', n_classes, lr, momentum, d['train'], y, val_data, batch_size, epochs)

if __name__=="__main__":
    test_network_4()