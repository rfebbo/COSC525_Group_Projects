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

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    #Extract mean and log of variance
    z_mean, z_log_var = args
    #get batch size and length of vector (size of latent space)
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    #Return sampled number (need to raise var to correct power)
    return z_mean + K.exp(z_log_var) * epsilon


def run_model_5(builder, name, lr, momentum, x, val_data, batch_size, epochs):

    print('running ' + name)
    model, z_log_var, z_mean, inputs, outputs = builder()

    #setting loss
    reconstruction_loss = keras.losses.mse(inputs, outputs)
    reconstruction_loss *=1
    kl_loss = K.exp(z_log_var) + K.square(z_mean) - z_log_var - 1
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= 0.001
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)
    model.compile(optimizer='adam')

    history=model.fit(x,validation_data=(val_data,None),batch_size=batch_size,epochs=epochs, verbose=True)

    pd.DataFrame.from_dict(history.history,orient='index').to_csv('./saved_runs/'+name + '(lr_' + str(lr) + ')(batch_' + str(batch_size) + ')(epoch_' + str(epochs) + ')' + '.csv')


def build_model_5():
    # model=Sequential()

    original_dim = (32,32,1)
    latent_dim = 8
    # encoder
    inputs = Input(shape = original_dim, name='encoder_input')
    x = layers.Conv2D(20,3,activation='relu', padding='valid')(inputs)
    x = layers.Conv2D(20,3,activation='relu', padding='valid')(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_output')

    #decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Reshape(target_shape=(2,2,2))(latent_inputs)
    x = layers.Conv2DTranspose(10,8,activation='relu', padding='valid')(x)
    x = layers.Conv2DTranspose(10,8,activation='relu', padding='valid')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(np.asarray(original_dim).size, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs, name='decoder_output')

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_cnn')
    return vae, z_log_var, z_mean, inputs, outputs


def test_network_5():
    d = read_data()

    # lrs = [0.05]
    lrs = [0.01, 0.1]
    momentum = 0.9
    batch_size =  128
    epochs = 100

    for lr in lrs:
        run_model_5(build_model_5, 'task_5', lr, momentum, d['train'], d['val'], batch_size, epochs)

if __name__=="__main__":
    test_network_5()