import os
from models import *
# os.environ['CUDA_VISIBLE_DEVICES']=""
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# print('tf version: ', tf.__version__)
import tensorflow as tf
from part1 import create_sequences
from part2 import createInputOutput
from tensorflow import keras
import pandas as pd
import numpy as np
from part3 import predict



class Predict(keras.callbacks.Callback):
    def __init__(self, model, x_train, uni, window_size, period, output_len, n_pred, temp):
        """ Save params in constructor
        """
        self.model = model
        self.x_train = x_train
        self.uni = uni
        self.window_size = window_size
        self.period = period
        self.output_len = output_len
        self.n_pred = n_pred
        self.temp = temp

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        if epoch % self.period == 0:
            for o in range(self.n_pred):
                # get random characters from x_train
                initial_characters = self.x_train[np.random.randint(0, len(self.x_train))].reshape(1,self.window_size, len(self.uni))

                decoded = predict(self.output_len,self.model,self.temp,self.window_size, initial_characters, self.uni, True)
                
                inp = '"'
                for c in decoded[0:self.window_size]:
                    inp += c
                inp += '"'

                outp = '"'
                for c in decoded[self.window_size:]:
                    outp += c
                outp += '"'
                print(f'input: {inp}, output: {outp}')


def train_model(model, name, x_train, y_train, h, epochs, window_size, stride, uni):

    model = model(h, len(uni))
    model.summary()

    #use default learning rate and momentum
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs,verbose=True,batch_size=64,callbacks=[Predict(model, x_train, uni, window_size, 5, 5, 2, 1)])
    model.save(f'saved_models/{name}')

    pd.DataFrame.from_dict(history.history,orient='index').to_csv(f'./saved_histories/{name}.csv')

    return model


if __name__=="__main__":
    # python3 rnn.py beatles.txt lstm 100 10 5 1
    # Will run the code with an LSTM with a hidden state of size 100 and a window size of10, a stride of 5 and a sampling temperature of 1.2

    if (len(os.sys.argv) == 2 and os.sys.argv[1] == 'train-all'):
        # window sizes and strides
        ws_s = [(5,3),(15,10)]
        models = [simple_model, simple_model_stacked, LSTM_model, LSTM_model_stacked]
        hidden_states = [100, 200]
        epochs = 40
        m_types = ['simple_model', 'simple_model_stacked', 'LSTM_model', 'LSTM_model_stacked']

        for m_type, m in zip(m_types, models):
            for h in hidden_states:
                for ws in ws_s:
                    name = m_type+f'(epochs:{epochs})(hidden:{h}perRNN_layer)(w:{ws[0]},s:{ws[1]})'
                    print(f'training model {name}')
                    uni, seq = create_sequences('beatles.txt',ws[0],ws[1])

                    x_train, y_train = createInputOutput(uni,seq)
                    train_model(m, name, x_train, y_train, h, epochs=epochs, window_size=ws[0], stride=ws[1], uni=uni)
    elif (len(os.sys.argv) == 8):

        file_name = os.sys.argv[1]
        m_type = os.sys.argv[2]
        h_size = int(os.sys.argv[3])
        window_size = int(os.sys.argv[4])
        stride = int(os.sys.argv[5])
        temperature = float(os.sys.argv[6])
        epochs = int(os.sys.argv[7])

        name = m_type+f'(epochs:{epochs})(hidden:{h_size}perRNN_layer)(w:{window_size},s:{stride})'

        uni, seq = create_sequences(file_name,window_size,stride)
        x_train, y_train = createInputOutput(uni,seq)

        if(m_type == 'lstm'):
            print('training lstm model with:',
                    f"\n\twindow size: {window_size}\n\tstride: {stride}\n\tepochs: {epochs}",
                    f"\n\tdata: {file_name}\n\thidden nodes: {h_size}\n\ttemperature:{temperature}")
            train_model(LSTM_model, name=name, x_train=x_train, y_train=y_train, h=h_size, epochs=epochs, window_size=window_size, stride=stride, uni=uni)
        elif(m_type == 'stacked_lstm'):
            print(f'training stacked lstm model with:',
                    f"\n\twindow size: {window_size}\n\tstride: {stride}\n\tepochs: {m_type}",
                    f"\n\tdata: {file_name}\n\thidden nodes: {h_size}\n\ttemperature:{temperature}")
            train_model(LSTM_model_stacked, name=name, x_train=x_train, y_train=y_train, h=h_size, epochs=epochs, window_size=window_size, stride=stride, uni=uni)
        elif(m_type == 'simpleRNN'):
            print(f'training simpleRNN model with:',
                    f"\n\twindow size: {window_size}\n\tstride: {stride}\n\tepochs: {m_type}",
                    f"\n\tdata: {file_name}\n\thidden nodes: {h_size}\n\ttemperature:{temperature}")
            train_model(simple_model, name=name, x_train=x_train, y_train=y_train, h=h_size, epochs=epochs, window_size=window_size, stride=stride, uni=uni)
        elif(m_type == 'stacked_simpleRNN'):
            print(f'training stacked simpleRNN model with:',
                    f"\n\twindow size: {window_size}\n\tstride: {stride}\n\tepochs: {m_type}",
                    f"\n\tdata: {file_name}\n\thidden nodes: {h_size}\n\ttemperature:{temperature}")
            train_model(simple_model_stacked, name=name, x_train=x_train, y_train=y_train, h=h_size, epochs=epochs, window_size=window_size, stride=stride, uni=uni)
        else:
            print(f"model {m_type} unknown. Please use (lstm, stacked_lstm, simpleRNN, stacked_simpleRNN).")
    else:
        print("usage: \n\tpython3 part4.py beatles.txt lstm 100 10 5 1 40\n",
                "Will run the code with an LSTM with a hidden state of size 100 and a window size of 10, a stride of 5 and a sampling temperature of 1 over 40 epochs\n",
                "usage: \n\tpython3 part4.py train-all\n Will run all 4 models with 40 epochs and (window_size, stride) of (5,3) and (15,10) and hidden states of 100 and 200(16 models total)")
    