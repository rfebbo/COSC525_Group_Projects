
import os
# os.environ['CUDA_VISIBLE_DEVICES']=""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, SimpleRNN


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print('physical_devices', physical_devices)

def simple_model(h, uni):
    model=Sequential()
    model.add(SimpleRNN(h,input_dim=uni, return_sequences=False))
    model.add(Dense(uni,activation='softmax'))
    return model


def simple_model_stacked(h, uni):
    model=Sequential()
    model.add(SimpleRNN(h,input_dim=uni, return_sequences=True))
    model.add(SimpleRNN(h, activation='relu'))
    model.add(Dense(uni,activation='softmax'))
    return model

def LSTM_model(h, uni):
    model=Sequential()
    model.add(LSTM(h,input_dim=uni))
    model.add(Dense(uni,activation='softmax'))
    return model

def LSTM_model_stacked(h, uni):
    model=Sequential()
    model.add(LSTM(h,input_dim=uni, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(h))
    model.add(Dense(uni,activation='softmax'))
    return model