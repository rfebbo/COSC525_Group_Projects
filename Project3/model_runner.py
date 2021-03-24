from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import pandas as pd
import tensorflow as tf

def run_model(builder, name, n_classes, lr, momentum, x, y, val_data, batch_size, epochs):

    print('running ' + name)
    sgd = optimizers.SGD(lr=lr,momentum=momentum)
    loss = tf.keras.losses.CategoricalCrossentropy()

    model = builder(n_classes)
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    history=model.fit(x,y,validation_data=val_data,batch_size=batch_size,epochs=epochs, verbose=True)

    pd.DataFrame.from_dict(history.history,orient='index').to_csv(name + '(lr_' + str(lr) + ')(batch_' + str(batch_size) + ')(epoch_' + str(epochs) + ')' + '.csv')

def run_all_models(builder, name, d, lr, momentum, batch_size, epochs):
    # run race network
    run_model(builder, name + '_race', len(d['race_classes']), lr, momentum, d['train'],d['race_t_labels'],(d['val'],d['race_v_labels']),batch_size,epochs)

    # run gender network
    run_model(builder, name + '_gender', len(d['gender_classes']), lr, momentum, d['train'],d['gender_t_labels'],(d['val'],d['gender_v_labels']),batch_size,epochs)

    # run age network   
    run_model(builder, name + '_age', len(d['age_classes']), lr, momentum, d['train'],d['age_t_labels'],(d['val'],d['age_v_labels']),batch_size,epochs)