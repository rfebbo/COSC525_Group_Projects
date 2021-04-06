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


from tensorflow.keras import backend as K

from read_data import read_data

import matplotlib.pyplot as plt

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

# vae_model = tf.keras.models.load_model('vae_model')
decoder_model = tf.keras.models.load_model('decoder_model')
decoder_model.summary()
pred = decoder_model.predict(np.random.rand(1,9))

for i in range(10):

    ax = plt.subplot(2,5,i+1)
    pred = decoder_model.predict(np.random.rand(1,9))
    pred = np.reshape(pred, (32,32,1))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.imshow(pred, cmap='Greys_r')
    # plt.title("Image " + str(i))
# print(pred)

plt.savefig("generated faces.png")