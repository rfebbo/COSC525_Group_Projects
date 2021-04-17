
import os
# os.environ['CUDA_VISIBLE_DEVICES']=""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('physical_devices', physical_devices)

unique_chars=50

def simple_model1():
    model=Sequential()
    model.add(SimpleRNN(100,input_dim=50))
    model.add(Dense(50))
    return model

def simple_model2():
    model=Sequential()
    model.add(SimpleRNN(200,input_dim=unique_chars))
    model.add(Dense(unique_chars))
    return model

def simple_model3():
    model=Sequential()
    model.add(SimpleRNN(100,input_dim=unique_chars, return_sequences=True))
    model.add(SimpleRNN(100, activation='relu'))
    model.add(Dense(unique_chars))
    return model

def LSTM_model1():
    model=Sequential()
    model.add(LSTM(100,input_dim=unique_chars))
    model.add(Dense(unique_chars))
    return model

def LSTM_model2():
    model=Sequential()
    model.add(LSTM(200,input_dim=unique_chars))
    model.add(Dense(unique_chars))
    return model

def LSTM_model3():
    model=Sequential()
    model.add(LSTM(100,input_dim=unique_chars, return_sequences=True))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(unique_chars))
    return model