from models import *
import tensorflow as tf
from part1 import create_sequences
from part2 import createInputOutput


uni, seq = create_sequences(3,5)
x_train, y_train = createInputOutput(uni,seq)


model = simple_model1(len(uni))
model.summary()

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
