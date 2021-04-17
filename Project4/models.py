
import os
# os.environ['CUDA_VISIBLE_DEVICES']=""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('physical_devices', physical_devices)

def simple_model1(uni):
    model=Sequential()
    model.add(SimpleRNN(100,input_dim=uni))
    model.add(Dense(uni,activation='softmax'))
    return model

def simple_model2(uni):
    model=Sequential()
    model.add(SimpleRNN(200,input_dim=uni))
    model.add(Dense(uni,activation='softmax'))
    return model

def simple_model3(uni):
    model=Sequential()
    model.add(SimpleRNN(100,input_dim=uni, return_sequences=True))
    model.add(SimpleRNN(100, activation='relu'))
    model.add(Dense(uni,activation='softmax'))
    return model

def LSTM_model1(uni):
    model=Sequential()
    model.add(LSTM(100,input_dim=uni))
    model.add(Dense(uni,activation='softmax'))
    return model

def LSTM_model2(uni):
    model=Sequential()
    model.add(LSTM(200,input_dim=uni))
    model.add(Dense(uni,activation='softmax'))
    return model

def LSTM_model3(uni):
    model=Sequential()
    model.add(LSTM(100,input_dim=uni, return_sequences=True))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(uni,activation='softmax'))
    return model