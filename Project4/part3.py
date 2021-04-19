import os
from models import *
# os.environ['CUDA_VISIBLE_DEVICES']=""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from part1 import create_sequences
import re

def atoi(text):
    return int(text) if text.isdigit() else text
    
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



def predict(num_char, model, temperature, window_size, initial_characters, uni, encoded_input):
    
    outputs = [initial_characters]

    if (encoded_input is False):
        encoded_init = np.zeros((1,len(initial_characters), len(uni)))
        for i, c in enumerate(initial_characters):
            for u, un in enumerate(uni):
                if (un == c):
                    encoded_init[0,i,u] = 1

        outputs = [encoded_init]

    
    for c in range(num_char):
        # print(outputs[c].shape)
        outputs.append(np.asarray(model.predict(outputs[c])).reshape(1,1,len(uni)))

    #add temperature and decode
    decoded = []

    for o in outputs:
        for c in o:
            for i in c:
                # print(i)
                # preds = np.log(i) / temperature
                # exp_preds = np.exp(preds)
                # preds = exp_preds / np.sum(exp_preds)
                preds=i
                if uni[np.argmax(preds)] == '=':
                    decoded.append('\n')
                else:
                    decoded.append(uni[np.argmax(preds)])
    

    return decoded



if __name__=="__main__":

    files = os.listdir('./saved_models/')
    files.sort(key=natural_keys)

    initial_input = 'she loves you, yea'

    temps = np.arange(1,5.2,1)
    print(temps)
    # exit()
    # temps = [1]

    for f in files:
        # f = 'stacked_lstm(epochs:20)(hidden:150perRNN_layer)(w:30,s:5)'
        print(f'model = {f}')
        for t in temps:
            m_type_end = f.find('(epochs')
            e_end = f.find(')(hidden:')
            h_end = f.find(')(w:')
            w_end = f.find(',s:')


            m_info = {}
            m_info['epochs'] = f[m_type_end+8:e_end]
            m_info['hidden'] = f[e_end+9:h_end]
            window_size = int(f[h_end+4:w_end].strip())
            stride = int(f[w_end+3:-1].strip())
            # print(f)
            # print(window_size)
            # print(len(window_size))
            # print(stride)
            # print(len(stride))
            # continue
            cur_input = initial_input[0:window_size]

            uni, seq = create_sequences('beatles.txt', window_size, stride)
            model = tf.keras.models.load_model('./saved_models/' + f)
            outputs = predict(50, model, t, window_size, cur_input, uni, False)
            with open('./saved_predictions/' + f + 'temp:'+str(np.round(t,2)), 'w') as pf:
                pf.write(f'input: {outputs[0:len(cur_input)]}')
                pf.write(f'output: {outputs[len(cur_input):]}')
            
            # print(f'input: {outputs[0:len(cur_input)]}')
            # print(f'output: {outputs[len(cur_input):]}')
